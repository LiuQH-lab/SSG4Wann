# SSG4Wann

A MPI-enabled tool for **symmetrizing Wannier tight-binding Model**
(`*_hr.dat`/`*_tb.dat`) generated from Wannier90, using the Oriented Spin
Space Group (OSSG) symmetry of the magnetic system and supporting both strong
and weak spin-orbit coupling limits.

For more details, please refer to the
[documentation](https://ssg4wann.readthedocs.io/en/latest/).

## Overview

`SSG4Wann` is designed to restore or enforce symmetry constraints on Wannier
Hamiltonians by averaging matrix elements under symmetry operations. It
supports both collinear (up/down channels) and non-collinear workflows and is
optimized for larger workloads through MPI-based parallel computation.

## Key Features

- Symmetrization of Wannier90 HR Hamiltonians with both MSG and SSG symmetries
  and support for spin channels and non-collinear settings
- MPI parallelization for efficient processing of large Hamiltonians
- Configurable behavior through `sg.in`
- Wannier90 HR and TB input and output
- Optional band transformation workflow controls

## Installation

SSG4Wann requires Python **3.12+**. It is available on PyPI:

```bash
python -m pip install ssg4wann
ssg4wann --version
```

To include MPI support:

```bash
python -m pip install "ssg4wann[mpi]"
```

## Quick Start

Prepare a working directory containing an `INCAR` file, the relevant
Wannier90 `.win` file, and a Wannier90 `*_hr.dat` or `*_tb.dat` file.

Generate an initial configuration and run the symmetrization:

```bash
cd /path/to/calculation
ssg4wann --init
ssg4wann -c sg.in
```

You can also use the Python API:

```python
import ssg4wann as sw

sw.quick_run()
```

> [!WARNING]
> A run that finishes without an exception is not necessarily physically
> correct. The spin direction, Wannier projection order, atomic positions,
> and Hamiltonian basis must be mutually consistent.

## Documentation

- [Getting Started](https://ssg4wann.readthedocs.io/en/latest/getting-started/)
  covers installation, input files, serial execution, and output files.
- [Configuration Reference](https://ssg4wann.readthedocs.io/en/latest/configuration/)
  documents the `sg.in` tags.
- [Basis, Spin and Symmetry Conventions](https://ssg4wann.readthedocs.io/en/latest/conventions/)
  describes the conventions required for a consistent calculation.
- [Troubleshooting and Compatibility](https://ssg4wann.readthedocs.io/en/latest/troubleshooting/)
  collects common errors and known VASP compatibility issues.
- [Examples](https://ssg4wann.readthedocs.io/en/latest/examples/) describes the
  calculations included in the repository.
- [MPI and HPC](https://ssg4wann.readthedocs.io/en/latest/mpi-hpc/) covers local
  MPI execution and cluster job scripts.

## License

This project is licensed under the Apache License, Version 2.0.
See the LICENSE file for details.
