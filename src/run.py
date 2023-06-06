#from mpi4py import MPI
import numpy as np
import ase
from ase import Atoms, Atom
from ase.build import bulk
from ase.calculators.vasp import Vasp
import lammps
from fitsnap3lib.scrapers.ase_funcs import ase_scraper
from fitsnap3lib.fitsnap import FitSnap
import random
import sys
import os
import random

def initialize(fit_settings, fs_uq_settings, load_fit=False):
    """
    Initialize fitting data and per-atom descriptors for UQ.
    """
    if load_fit:
        atot = np.load("atot.npy")
        btot = np.load("btot.npy")
        wtot = np.load("wtot.npy")
        d_all = np.load("dall.npy")

    else:

        fs = FitSnap(fit_settings, arglist=["--overwrite"])
        fs2 = FitSnap(fs_uq_settings,arglist=["--overwrite"])

        # Get data from ASE list
        from ase.io import read
        filenames = ["training-data/md_300K.xyz"]
        frames = []
        for filename in filenames:
            frames.extend(read(filename, ":"))

        # Shorten frames, only use 2 for starting out.
        frames = random.choices(frames, k=2)

        print(f"Fitting to {len(frames)} configs.")

        # Convert ASE to fitsnap data structure
        from fitsnap3lib.scrapers.ase_funcs import ase_scraper
        data = ase_scraper(frames)
        print(type(data))

        # Feed data into descriptor calculation.
        fs.process_configs(data=data)

        # Perform fit.
        fs.solver.perform_fit()

        # Write LAMMPS file.
        fs.output.write_lammps(fs.solver.fit)

        # Extract covariance matrix, used to get uncertainties of future structures.
        #print(np.shape(fs.solver.cov))
        #cov = fs.solver.cov # Shape (num_desc, num_desc)
        atot = fs.pt.shared_arrays['a'].array # Shape (nrows, num_desc)
        btot = fs.pt.shared_arrays['b'].array
        wtot = fs.pt.shared_arrays['w'].array

        descriptors_list = []
        for i, configuration in enumerate(data):
            a,b,w = fs2.calculator.process_single(configuration)
            descriptors_list.append(a)
        d_all = np.concatenate(descriptors_list, axis=0)

        # Set load_fit to True so that we load data when restarting the Python script.

    return atot, btot, wtot, d_all


def add2set(lmp, fs, d_set):
    """
    Calculate UQ metric per atom.
    NOTE: num_coeff is often num_desc + 1, hence the indexing seen below.

    Args:
        lmp: lammps instance containing a descriptor compute.
        fs: fitsnap instance containing UQ metrics.
        d_set: optional per-atom descriptors to compare against.
    """

    natoms = lmp.get_natoms()
    lmp_snap = extract_compute_np(lmp, "snap", 0, 2, None)
    
    # Get peratom descriptors of this configuration.
    d_m = lmp_snap[0:natoms,0:-1]

    # Concatenate to d_set (modify in place).
    d_set = np.concatenate((d_set, d_m), axis=0)

    return d_set

def calc_uq_peratom(lmp, fs, d_set=None):
    """
    Calculate UQ metric per atom.
    NOTE: num_coeff is often num_desc + 1, hence the indexing seen below.

    Args:
        lmp: lammps instance containing a descriptor compute.
        fs: fitsnap instance containing UQ metrics.
        d_set: optional per-atom descriptors to compare against.
    """

    natoms = lmp.get_natoms()
    lmp_snap = extract_compute_np(lmp, "snap", 0, 2, None)
    
    # Get peratom descriptors of this configuration.
    d_m = lmp_snap[0:natoms,0:-1]

    #print(np.shape(d_m))
    #print(np.shape(d_set))

    uq_peratom = []
    for di in d_m:
        d = np.array([di])
        dist = np.linalg.norm(d_set - d, axis=1)
        #mean_dist = np.mean(dist)
        #uq_peratom.append(mean_dist)
        min_dist = np.min(dist)
        uq_peratom.append(min_dist)
    uq_peratom = np.array(uq_peratom)

    # Concatenate to d_set (modify in place).
    d_set = np.concatenate((d_set, d_m), axis=0)

    return uq_peratom

def write_xyz_uq(fh, lmp, uq):
    """
    Write XYZ file with per atom UQ property; mainly for visualization purposes.
    """

    # Write XYZ file for visualization.
    # Args: uq_peratom, lmp, fs
    natoms = lmp.get_natoms()

    # Convert to 3x3 cell for ASE/VASP/Ovito compatibility.
    box = lmp.extract_box()
    boxlo = box[0]
    boxhi = box[1]
    cell = np.zeros((3,3))
    cell[0,0] = boxhi[0]-boxlo[0]
    cell[1,1] = boxhi[1]-boxlo[1]
    cell[2,2] = boxhi[2]-boxlo[2]
    lattice_str = f'Lattice="{cell[0,0]} {cell[0,1]} {cell[0,2]} {cell[1,0]} {cell[1,1]} {cell[1,2]} {cell[2,0]} {cell[2,1]} {cell[2,2]}"'

    lmp_pos = lmp.numpy.extract_atom_darray(name="x", nelem=natoms, dim=3)
    lmp_types = lmp.numpy.extract_atom_iarray(name="type", nelem=natoms).ravel()

    fh.write(f"{natoms}\n")
    line = f"{lattice_str} Properties=type:S:1:pos:R:3:uq:R:1\n"
    fh.write(line)
    for i in range(0,natoms):
        t = lmp_types[i]
        line = f"{t} {lmp_pos[i,0]} {lmp_pos[i,1]} {lmp_pos[i,2]} {uq[i]}\n"
        fh.write(line)

def lmp2atoms(lmp):
    """
    Convert lmp instance to ASE atoms object.

    Args:
        lmp: a lammps instance.
    
    Returns an ASE atoms object.
    """
    natoms = lmp.get_natoms()

    # Convert to 3x3 cell for ASE/VASP/Ovito compatibility.
    box = lmp.extract_box()
    boxlo = box[0]
    boxhi = box[1]
    cell = np.zeros((3,3))
    cell[0,0] = boxhi[0]-boxlo[0]
    cell[1,1] = boxhi[1]-boxlo[1]
    cell[2,2] = boxhi[2]-boxlo[2]

    atoms = Atoms(f'Si{natoms}', \
                  positions=lmp.numpy.extract_atom_darray(name="x", nelem=natoms, dim=3), \
                  cell=cell, \
                  pbc=[1,1,1])

    return atoms

def extract_compute_np(lmp, name, compute_style, result_type, array_shape=None):
    """
    Convert a lammps compute to a numpy array.
    Assumes the compute stores floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is
    skipped and a python float is returned.
    From LAMMPS/src/library.cpp:
    style = 0 for global data, 1 for per-atom data, 2 for local data
    type = 0 for scalar, 1 for vector, 2 for array
    """

    if array_shape is None:
        array_np = lmp.numpy.extract_compute(name,compute_style, result_type)
    else:
        ptr = lmp.extract_compute(name, compute_style, result_type)
        if result_type == 0:

            # no casting needed, lammps.py already works

            return ptr
        if result_type == 2:
            ptr = ptr.contents
        total_size = np.prod(array_shape)
        buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double * total_size))
        array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
        array_np.shape = array_shape
    return array_np

def run_vasp(atoms):

    mydir = '.'    # Directory where we will do the calculations

    # Make self-consistent ground state
    kpts = ase.dft.kpoints.monkhorst_pack([4,4,4])
    calc = Vasp(directory=mydir)
    calc.set(ismear=0,
             isif=2,
             istart=0,
             algo="fast",
             kpts=kpts)

    atoms.calc = calc
    atoms.get_potential_energy()  # Run the calculation
    forces = atoms.get_forces()

    frames = [atoms]
    data = ase_scraper(frames)

    # Hard-code virial weight to be 0.1 for now... can make option later.
    data[0]['vweight'] = 1e-2

    # Convert stress units from VASP (kbar) to LAMMPS metal units (bar).
    data[0]['Stress'] *= 1e3

    return data
