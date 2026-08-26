# SSG4Wann

SSG4Wann is an MPI-enabled Python tool for symmetrizing Wannier90
tight-binding Hamiltonians using the symmetry of an oriented spin space group
(OSSG). It supports collinear and non-collinear workflows, calculations with
and without spin-orbit coupling, and Wannier90 HR and TB files.

## What SSG4Wann does

SSG4Wann restores or enforces symmetry constraints on a real-space Wannier
Hamiltonian by averaging its matrix elements under the relevant symmetry
operations.

When TB mode is enabled, SSG4Wann also symmetrizes the Cartesian position
matrix as a vector operator.

## Key features

- Full OSSG symmetrization without spin-orbit coupling.
- MSG symmetrization for calculations with spin-orbit coupling.
- Collinear up/down and non-collinear spinor Hamiltonians.
- Wannier90 HR and TB input and output.
- MPI parallelization for larger models and HPC environments.
- Optional band-structure data generation.
- Automatic generation of an initial `sg.in` from VASP and Wannier90 inputs.

## Installation

SSG4Wann requires Python 3.12 or later. Install the current release from PyPI:

```bash
python -m pip install ssg4wann
```

Confirm the installation:

```bash
ssg4wann --version
ssg4wann --help
```

## Quick start

Prepare a working directory containing an `INCAR` file, the relevant
Wannier90 `.win` file, and a Wannier90 `*_hr.dat` or `*_tb.dat` file.

```bash
cd /path/to/calculation
ssg4wann --init
ssg4wann -c sg.in
```

!!! warning

    A run that finishes without an exception is not necessarily physically
    correct. The spin direction, Wannier projection order, atomic positions,
    and Hamiltonian basis must be mutually consistent.

## Documentation

- [Getting Started](getting-started.md) covers installation, required input
  files, serial execution, and output files.
- [Configuration Reference](configuration.md) documents the `sg.in` tags.
- [Basis, Spin and Symmetry Conventions](conventions.md) describes the
  conventions required for a consistent calculation.
- [Troubleshooting and Compatibility](troubleshooting.md) collects common
  errors and known VASP compatibility issues.
- [Examples](examples.md) describes the Fe, Fe with SOC, Fe TB, and Nb3VS6
  calculations included in the repository.
- [MPI and HPC](mpi-hpc.md) covers local MPI execution and Slurm or LSF job
  templates.

## Project links

- [GitHub repository](https://github.com/LiuQH-lab/SSG4Wann)
- [PyPI project](https://pypi.org/project/ssg4wann/)

## License

SSG4Wann is distributed under the Apache License 2.0.
