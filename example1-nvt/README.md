# Active Learning Instructions

1. Install LAMMPS serially to use this script, since we need MPI for the VASP ASE call.

2. Install FitSNAP (clone and set `PYTHONPATH` or `python -m pip install fitsnap3`).

3. Set the following environment variables for ASE and VASP:

        # Choose nodes and cores.
        export nodes=1
        export cores=36
        # VASP command.
        export ASE_VASP_COMMAND="mpiexec --bind-to core --npernode $cores --n $(($nodes*$cores)) /path/to/bin/vasp_std"
        export VASP_PP_PATH=./ # same directory that `potpaw` dir containing pseudopotential is in

4. Export `active-learning` directory to `PYTHONPATH`.

5. `python run.py`