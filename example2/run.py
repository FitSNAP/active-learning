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

# Function for initializing the active learning cycle.
from src2.run import initialize
# Function for generating structures.
from structure_generator import run_md

# Compression cycle settings.
nsteps = 10000
nchecks = 1000
md_settings = \
{
"nsteps": 1000, # Number of steps for MD.
"nchecks": 100, # Number of checks per cycle.
"ncheck_every": int(nsteps/nchecks), # Check UQ every this many steps.
"velocity_seed": random.randint(1,1e6) # For LAMMPS velocities.
}

# FitSNAP settings for the fitting the potential.
fit_settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Si",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 1
    },
"ESHIFT":
    {
    "Si": 0.0
    },
"SOLVER":
    {
    "solver": "SVD"
    },
"OUTFILE":
    {
    "metrics": "Si_metrics.md",
    "potential": "Si_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 0.9 2.0",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 10 10"
    }
}

# FitSNAP settings for calculating peratom descriptors for UQ.
fs_uq_settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Si",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 1,
    "quadraticflag": 0,
    "bikflag": 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 0,
    "stress": 0
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "zero 10.0",
    "pair_coeff": "* *"
    }
}

# Fit to 1 configuration to get an intitial potential.
atot, btot, wtot, d_all = initialize(fit_settings, fs_uq_settings, load_fit=False)

# Loop through structure generator and take top uncertain.
frames = run_md(d_all, md_settings)
print(f"Got {len(frames)} structures from generator.")

# Feed these frames into GPAW.
# ...

# Collect Atoms objects.
# ...

# Feed back into fit.
# ...



