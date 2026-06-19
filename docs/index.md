# SSG4Wann

SSG4Wann is an MPI-enabled Python tool for symmetrizing Wannier90
tight-binding Hamiltonians using the symmetry of an oriented spin space group
(OSSG). It supports collinear and non-collinear workflows, calculations with
and without spin-orbit coupling, and Wannier90 HR and TB files.

## What SSG4Wann does

SSG4Wann restores or enforces symmetry constraints on a real-space Wannier
Hamiltonian by averaging its matrix elements under the relevant symmetry
operations. The main workflow is:

1. Read the calculation settings from `sg.in`.
2. Load a Wannier90 `*_hr.dat` or `*_tb.dat` model.
3. Read the lattice, Wannier centers, and projection order from the `.win`
   file.
4. Obtain the magnetic symmetry operations with `findspingroup`.
5. Construct their orbital and spin representations.
6. Symmetrize the Hamiltonian and write a new Wannier90-compatible model.

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

To include MPI support:

```bash
python -m pip install "ssg4wann[mpi]"
```

Confirm the installation:

```bash
ssg4wann --version
ssg4wann --help
```

## Quick start

Prepare a working directory containing:

- An `INCAR` file.
- The relevant Wannier90 `.win` file.
- A Wannier90 `*_hr.dat` or `*_tb.dat` file.

Generate an initial configuration:

```bash
cd /path/to/calculation
ssg4wann --init
```

Review the generated `sg.in`, especially `soc`, `NONCOLLINEAR_channel`,
`spin_direction`, and the Wannier basis order. Then run:

```bash
ssg4wann -c sg.in
```

Alternatively, specify the working directory explicitly:

```bash
ssg4wann -c sg.in -w /path/to/calculation
```

!!! warning

    A run that finishes without an exception is not necessarily physically
    correct. The spin direction, Wannier projection order, atomic positions,
    and Hamiltonian basis must be mutually consistent.

## Choosing the symmetry

The `soc` setting selects the operation set used by SSG4Wann:

```ini
# Full oriented spin space group
soc = False

# Magnetic space group subgroup
soc = True
```

Thus, `soc = False` is used for the full OSSG workflow without spin-orbit
coupling, while `soc = True` selects the corresponding MSG operations for the
spin-orbit-coupled workflow.

## Next steps

- See [Examples](examples.md) for the Fe, Fe with SOC, Fe TB, and Nb3VS6
  calculations included in the repository.
- See [MPI and HPC](mpi-hpc.md) for local MPI execution and Slurm or LSF job
  templates.
- Visit the
  [GitHub repository](https://github.com/LiuQH-lab/SSG4Wann) to report an
  issue or inspect the source code.
- Visit the [PyPI project](https://pypi.org/project/ssg4wann/) for published
  releases.

## License

SSG4Wann is distributed under the Apache License 2.0.
