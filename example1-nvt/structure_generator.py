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
import lammps

# Function for adding current strucure features to pool of features.
from src.run import add2set
# Function for calculating UQ metric per atom.
from src.run import calc_uq_peratom
# Function for writing XYZ files with per atom UQ metric for visualization.
from src.run import write_xyz_uq
# Function for converting LAMMPS instance info to ASE object.
from src.run import lmp2atoms
# Function for running VASP.
from src.run import run_vasp

def compression_cycle(atot, btot, wtot, d_all, md_settings, fit_settings, uq_settings):

    # Get compression cycle settings.
    nsteps = md_settings["nsteps"]
    nchecks = md_settings["nchecks"]
    ncheck_every = md_settings["ncheck_every"]
    assert(ncheck_every == int(nsteps/nchecks))
    velocity_seed = md_settings["velocity_seed"]
    fh_log = md_settings["logfile"]

    # Get UQ settings.
    uq_threshold = uq_settings["uq_threshold"]
    uq_decrease = uq_settings["uq_decrease"]
    threshold_limit = uq_settings["threshold_limit"]
    fs_uq_settings = uq_settings["fs_uq_settings"]

    fs = FitSnap(fit_settings, arglist=["--overwrite"])
    fs2 = FitSnap(fs_uq_settings,arglist=["--overwrite"])

    lmp = lammps.lammps(cmdargs=["-screen", "lmp_screen.out"])

    lmp_setup=f"""
    # Initialize simulation

    variable a equal 5.431
    units           metal

    # generate the box and atom positions using a BCC lattice

    boundary        p p p

    lattice         diamond $a
    region          box block 0 1 0 1 0 1
    create_box      1 box
    create_atoms    1 box

    mass 1 28

    # choose potential

    include Si_pot.mod

    # Setup output

    thermo          100
    thermo_modify norm yes
    thermo_style custom step temp pe etotal press vol

    # Declare computes
    # bikflag 1 is necessary!
    compute snap all snap 4.67637 0.99363 6 0.5 1.0 rmin0 0.0 bzeroflag 1 quadraticflag 0 switchflag 1 bnormflag 0 wselfallflag 0 bikflag 1

    # Set up NVE run

    timestep 0.5e-3
    neighbor 1.0 bin
    neigh_modify once no every 1 delay 0 check yes
    velocity all create 600.0 {velocity_seed} loop geom
    #fix 1 all nve
    """

    lmp.command("clear")
    lmp.commands_string(lmp_setup)
    # Set up MD generator "process".
    #lmp.command("fix 1 all npt temp 2000.0 5000.0 0.05 iso 0.0 0.0 0.5")
    #lmp.command("fix 1 all npt temp 300 300 1 y 0 0 1 z 0 0 1 drag 1")

    # NVT works well with uniaxial compression cycles:
    lmp.command("fix 1 all nvt temp 300 300 0.05")

    # NVE can use temperature as an indicator:
    #lmp.command("fix 1 all nve")

    # Begin uniaxial compression cycle process.
    fh = open("dump_uq.xyz", 'w')

    count_threshold = 0
    stopstep = nsteps # This will tell LAMMPS when the end of the simulation is, we will do operations before then.

    box = lmp.extract_box()
    lengths = box[1]
    xlength = lengths[0]
    xlength_start = xlength
    # Computes and runs.
    lmp.command("uncompute snap")
    lmp.command("compute snap all snap 4.67637 0.99363 6 0.5 1.0 rmin0 0.0 bzeroflag 1 quadraticflag 0 switchflag 1 bnormflag 0 wselfallflag 0 bikflag 1")
    lmp.command(f"run 0 stop {stopstep}")
    uq = calc_uq_peratom(lmp, fs, d_all)
    write_xyz_uq(fh, lmp, uq)
    lmp.command("uncompute snap_press")

    #num_checks = 100
    for checks in range(nchecks):

        # Get thermo data.
        step = int(lmp.get_thermo('step'))
        pe = lmp.get_thermo('pe')
        temperature = lmp.get_thermo('temp')
        vol = lmp.get_thermo('vol')
        press = lmp.get_thermo('press')
        print(f" {step} {pe:0.2e} {temperature:0.2e} {vol:0.2f} {press:0.2e} {max(uq)}")
        fh_log.write(f" {step} {pe:0.2e} {temperature:0.2e} {vol:0.2f} {press:0.2e} {max(uq)}\n")

        # Computes and runs.
        # Need these extra commands to run multiple loops (uncompute snap, uncompute snap_press, etc.)
        lmp.command("uncompute snap")
        # Declare computes
        # bikflag 1 is necessary!
        lmp.command("compute snap all snap 4.67637 0.99363 6 0.5 1.0 rmin0 0.0 bzeroflag 1 quadraticflag 0 switchflag 1 bnormflag 0 wselfallflag 0 bikflag 1")
        lmp.command(f"run {ncheck_every} stop {stopstep}")

        # Get UQ metric per atom.
        uq = calc_uq_peratom(lmp, fs, d_all)

        # Write position and UQ data to XYZ file.
        write_xyz_uq(fh, lmp, uq)

        # Use UQ metric to decide if run VASP job.
        # NOTE: Anything could be a UQ metric - temperature, etc.
        #       - Min. atom distance is a good metric!
        #       - Some sort of geometric function penalizing things like trimers are good metrics!
        if (max(uq) > uq_threshold):
            count_threshold += 1
            uq_threshold *= uq_decrease
            print(f">>> Indicator threshold {count_threshold} reached; running DFT.")
            atoms = lmp2atoms(lmp)
            # Run VASP with atoms object to make fitsnap data.
            data = run_vasp(atoms)
            # Inject this data into descriptor calculation.
            am,bm,wm = fs.calculator.process_single(data[0])

            atot = np.concatenate((atot, am), axis=0)
            btot = np.concatenate((btot, bm), axis=0)
            wtot = np.concatenate((wtot, wm), axis=0)

            # Perform fit.
            fs.solver.perform_fit(atot, btot, wtot, trainall=True)
            # Write LAMMPS file.
            fs.output.write_lammps(fs.solver.fit)

            # Add this configuration to `d_all`.
            d_all = add2set(lmp, fs, d_all)

            # Save matrices.
            np.save("atot", atot)
            np.save("btot", btot)
            np.save("wtot", wtot)
            np.save("dall", d_all)

            print(np.shape(d_all))

            # Use new potential.
            lmp.command("include Si_pot.mod")

            if count_threshold >= threshold_limit:
                print(f"{count_threshold} thresholds reached; terminating.")
                return atot, btot, wtot, d_all

        # Uncompute snap_press to continue to next loop.
        lmp.command("uncompute snap_press")

    fh_log.close()