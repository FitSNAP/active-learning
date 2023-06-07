import numpy as np
from fitsnap3lib.scrapers.ase_funcs import ase_scraper
from fitsnap3lib.fitsnap import FitSnap
import lammps

# Function for calculating UQ metric per atom.
from src2.run import calc_uq_peratom
# Function for converting LAMMPS instance info to ASE object.
from src2.run import lmp2atoms

def run_md(d_all, md_settings, uq_settings):

    # Get compression cycle settings.
    nsteps = md_settings["nsteps"]
    nchecks = md_settings["nchecks"]
    ncheck_every = md_settings["ncheck_every"]
    assert(ncheck_every == int(nsteps/nchecks))
    velocity_seed = md_settings["velocity_seed"]

    # Get UQ settings.
    uq_threshold = uq_settings["uq_threshold"]
    threshold_limit = uq_settings["threshold_limit"]

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

    mass 1 180.88

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

    # NVT works well with uniaxial compression cycles:
    lmp.command("fix 1 all nvt temp 300 300 0.05")
    
    stopstep = nsteps # initially 

    lmp.command("uncompute snap")
    lmp.command("compute snap all snap 4.67637 0.99363 6 0.5 1.0 rmin0 0.0 bzeroflag 1 quadraticflag 0 switchflag 1 bnormflag 0 wselfallflag 0 bikflag 1")
    lmp.command(f"run 0 stop {stopstep}")
    uq = calc_uq_peratom(lmp, d_all)
    lmp.command("uncompute snap_press")

    structures = []
    max_uqs = []
    for checks in range(nchecks):

        # Get thermo data.
        step = int(lmp.get_thermo('step'))
        pe = lmp.get_thermo('pe')
        temperature = lmp.get_thermo('temp')
        vol = lmp.get_thermo('vol')
        press = lmp.get_thermo('press')
        print(f" {step} {pe:0.2e} {temperature:0.2e} {vol:0.2f} {press:0.2e} {max(uq)}")

        # Need these extra commands to run multiple loops (uncompute snap, uncompute snap_press, etc.)
        lmp.command("uncompute snap")
        # Declare computes
        # bikflag 1 is necessary!
        lmp.command("compute snap all snap 4.67637 0.99363 6 0.5 1.0 rmin0 0.0 bzeroflag 1 quadraticflag 0 switchflag 1 bnormflag 0 wselfallflag 0 bikflag 1")
        lmp.command(f"run {ncheck_every} stop {stopstep}")

        # Get UQ metric per atom.
        uq = calc_uq_peratom(lmp, d_all)

        atoms = lmp2atoms(lmp)
        structures.append(atoms)
        max_uqs.append(max(uq))

        # Uncompute snap_press to continue to next loop.
        lmp.command("uncompute snap_press")

    # Get 5 most uncertain structures.
    uq_argsort = np.argsort(max_uqs)[::-1]
    uq_sort = np.sort(max_uqs)[::-1]
    ret = [structures[uq_argsort[i]] for i in range(5)]
    return ret