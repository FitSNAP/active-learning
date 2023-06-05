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
from src.run import initialize
# Function for generating structures.
from structure_generator import compression_cycle

# Compression cycle settings.
nsteps = 10000
nchecks = 1000
cc_settings = \
{
"ncycles": 100, # Number of compression/decompression cycles.
"nsteps": 10000, # Number of steps in a half cycle.
"nchecks": 1000, # Number of checks per cycle.
"ncheck_every": int(nsteps/nchecks), # Check UQ every this many steps.
"compress_ratio": 0.9, # Ratio to compress.
"velocity_seed": random.randint(1,1e6), # For LAMMPS velocities.
"logfile": open("log1.out", 'w')
}

# UQ settings.
uq_settings = \
{
"uq_threshold": 5.0, # Some number to tell us when to do a DFT calculation.
"uq_decrease": 1.0, # factor by which to change UQ threshold each time DFT calculation is ran.
"threshold_limit": 2 # Number of thresholds reached before we restart the run.
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
    "stress": 0
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
    "pair_style": "hybrid/overlay zero 10.0 zbl 0.9 3.0",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 10 10"
    }
}

# FitSNAP settings for calculating peratom descriptors for UQ.
uq_settings["fs_uq_settings"] = \
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

# Other settings for this script only.
atot, btot, wtot, d_all = initialize(fit_settings, uq_settings["fs_uq_settings"], load_fit=False)

# Loop through structure generator to update fitting arrays and descriptor pool.
for run in range(1000):
    print(f">>> run: {run}")
    atot, btot, wtot, d_all = compression_cycle(atot, btot, wtot, d_all, cc_settings, fit_settings, uq_settings)