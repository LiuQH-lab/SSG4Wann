# MPI and HPC

SSG4Wann can distribute symmetry-operation processing, real-space
symmetrization, and band calculations across MPI processes. Serial execution
remains available when `mpi4py` is not installed or when the program is
launched with only one process.

## Install MPI support

An MPI-enabled installation requires:

1. An MPI implementation, such as Open MPI, MPICH, or Intel MPI.
2. `mpi4py` built for that MPI implementation.
3. SSG4Wann installed in the same Python environment.

For a local environment, install the optional MPI dependency with:

```bash
python -m pip install "ssg4wann[mpi]"
```

Check that the MPI launcher and Python binding are available:

```bash
module load mpi
mpirun --version
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

## Local parallel run

Prepare the same input files used for a serial run, then launch SSG4Wann
through MPI:

```bash
cd /path/to/calculation
mpirun -np 4 ssg4wann -c sg.in
```

You can also specify the working directory explicitly:

```bash
mpirun -np 4 ssg4wann -c sg.in -w /path/to/calculation
```

At startup, rank 0 reports the total number of MPI processes. Output and
progress messages are primarily written by rank 0.

SSG4Wann sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` to `1` to avoid unintended nested parallelism inside
each MPI process.

## Installing on an HPC cluster

On a cluster, first load the MPI implementation that will be used when the job
runs. Then create and activate the Python environment:

```bash
module load mpi
python -m venv /path/to/venv
source /path/to/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ssg4wann
```

Pre-built `mpi4py` wheels may use an MPI runtime that is incompatible with the
cluster environment. If necessary, build `mpi4py` with the cluster's MPI
compiler:

```bash
MPICC=mpicc python -m pip install --no-binary=mpi4py mpi4py
```

Verify the environment on a login or interactive compute node:

```bash
which python
which mpirun
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

The MPI library reported by `mpi4py` should match the MPI module used by the
launcher.

## Slurm example

The exact module names, account, partition, and MPI launch options depend on
the cluster. A minimal Slurm script is:

```bash
#!/bin/bash
#SBATCH --job-name ssg4wann
#SBATCH --partition 256G56c
#SBATCH --exclusive   
#SBATCH --ntasks 56  

module load intel/2022.0.1
module load mpi/2021.6.0
module purge
module load mpi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssg4wann_env

cd /path/to/calculation
mpirun -np "${SLURM_NTASKS}" ssg4wann -c sg.in
```

Some Slurm systems require `srun` instead of `mpirun`:

```bash
srun ssg4wann -c sg.in
```

Use the launch command recommended by the cluster administrators because the
required Slurm MPI plugin varies between systems.

## LSF example

A minimal LSF script is:

```bash
#!/bin/bash
#BSUB -J ssg4wann
#BSUB -q short
#BSUB -n 40
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -R "span[ptile=40]"

module purge
module load mpi/intel/2020.4

source ~/.bashrc
conda activate ssg4wann_env

cd /path/to/calculation
mpirun  -np 40 ssg4wann -c sg.in > $LSB_JOBID.log 2>&1
```

Submit it with:

```bash
bsub < job.lsf
```

## Choosing the number of processes

More MPI processes do not always make a calculation faster. The useful process
count is limited by the number of symmetry operations, real-space lattice
vectors, or k-points available in the current stage of the calculation.

Start with a small process count and measure both runtime and memory:

```bash
mpirun -np 8 ssg4wann -c sg.in
```

Each process holds calculation data, so increasing the process count can
increase total memory consumption. For large Wannier models, fewer processes
may be safer than using every available CPU core.

the predicted total memory use can be estimated with:

$$
MEM_{total} = n_{processes} \times MEM_{hr\  or \ tb} \times 2 
$$

As a practical starting point:

- Use 4-8 processes for small Fe-like examples.
- Increase gradually for larger real-space models.
- Request enough memory per node before increasing the task count.


## Runtime behavior and failures

SSG4Wann detects Slurm, PBS, and LSF environments and uses cluster-friendly
progress output. If one MPI rank raises an unhandled exception, the program
reports the failing rank and aborts the MPI communicator so that other ranks
do not remain waiting indefinitely.

For sufficiently large MPI result payloads, SSG4Wann can switch to a
file-buffered collection path to reduce the risk of oversized MPI messages.
This does not eliminate the need to request enough memory and local temporary
storage for the job.

## Common MPI problems

### `mpirun` starts multiple serial calculations

Confirm that `mpi4py` is installed in the same Python environment as
SSG4Wann:

```bash
which ssg4wann
which python
python -c "import ssg4wann, mpi4py; print(ssg4wann.__file__)"
```

### MPI library or symbol errors

This usually means that `mpi4py` was built against a different MPI
implementation. Reload the intended MPI module and rebuild `mpi4py` with its
`mpicc`.

### Job is killed because of memory use

Reduce the MPI process count or request more memory. Increasing the number of
processes can increase total memory use even when the work is distributed.

### The job hangs after one rank fails

Check the complete scheduler output for the first Python traceback and the
reported failing rank. The first error is normally more useful than later MPI
shutdown messages.

### Poor parallel speedup

Possible reasons include:

- The example is too small for MPI overhead to be worthwhile.
- There are fewer tasks than MPI processes in one or more workflow stages.
- File-system or communication overhead dominates the calculation.
- The job spans multiple nodes when a single node would be faster.

Benchmark the calculation with several process counts instead of assuming that
the largest allocation is best.
