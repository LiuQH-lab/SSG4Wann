# Getting Started

## Requirements

- Python **3.12+** (3.14.3 recommended)
- `numpy`
- `pandas`
- `tqdm`
- `findspingroup`
- `scipy`

Optional but recommended:

- `mpi4py`
- An MPI implementation:
  - OpenMPI, MPICH, Intel MPI, etc.

## Installation

It is available on PyPI, and you can install it with pip:

```bash
pip install ssg4wann
pip install ssg4wann --upgrade
ssg4wann --version
```

For more help, in the Command Line Interface (CLI), you can run:

```bash
ssg4wann --help
```

Python `>= 3.12` is required.

To include MPI support:

```bash
python -m pip install "ssg4wann[mpi]"
```

See [MPI and HPC](mpi-hpc.md) for local parallel execution and cluster
installation.

## Local minimum serial run example

Ensure the following files are prepared in your working directory:

- hr file(s)
- win file(s)
- INCAR file

### Python API example

```python
import ssg4wann as sw
sw.quick_run()
```

Also you can set the working directory and the config file path in the
`quick_run` function:

```python
import ssg4wann as sw
sw.quick_run(workdir="path/to/your/workdir", config_name="path/to/your/sg.in")
```

### Command line example

First, generate the `sg.in` file with the `--init` flag:

```bash
cd path/to/your/workdir
ssg4wann --init
# or you can specify the directory
ssg4wann --init -w path/to/your/workdir
```

You can directly run the code no matter there is a `sg.in` file or not, the
code will automatically generate one if it does not exist and run the
symmetrization with the generated `sg.in`. But you need to ensure that the
generated `sg.in` is correct according to the warnings. Or you can specify the
config file path with the `-c` flag and the working directory with the `-w`
flag:

```bash
# directly run
ssg4wann
# run with the specified config file and working directory
ssg4wann -c config.in -w path/to/your/workdir
```

## Input Files

At minimum, prepare:

1. **Wannier Hamiltonian file(s)**

    Depending on your channel mode:

    Non-collinear: `wannier90_hr.dat`

    Collinear: `wannier90.up_hr.dat` and `wannier90.dn_hr.dat`

2. **Wannier metadata files**

    Non-collinear: `wannier90.win`

    Collinear: `wannier90.up.win` and `wannier90.dn.win`

    The code will read the necessary Wannier basis, lattice structure,
    projection information from the `.win` file(s).

3. **INCAR file**

    The code will read `MAGMOM` in the INCAR file to determine the magnetic
    structure of the system, which is necessary for the correct symmetrization
    of the Hamiltonian. For collinear systems, the `MAGMOM` should be set to a
    single value per atom, while for non-collinear systems, the `MAGMOM` should
    be set to three values (x, y, z) per atom to specify the spin direction.

    If `ssg4wann` is going to generate the `sg.in` automatically, it will read
    `LNONCOLLINEAR` and `LSORBIT` tags in the INCAR file to determine the `soc`
    and `NONCOLLINEAR_channel` settings in the generated `sg.in`.

Furthermore, you can provide an optional `sg.in` file to specify the
configuration for the symmetrization process. If it is not provided, the code
will automatically generate one based on the input files and the system
parameters it detects. See the [Configuration Reference](configuration.md) for
details on the `sg.in` configuration.

## Output Files

Typical output includes symmetrized HR files or band structure data, depending
on the configuration:

- `*_symmed_hr.dat` (also produced from TB mode when `output_hr_from_tb = True`)
- `*_symmed_tb.dat`
- `*_band.dat`

Output naming is controlled by seed/config and channel logic in the code.

## Next steps

- Review every available `sg.in` option in the
  [Configuration Reference](configuration.md).
- Read [Basis, Spin and Symmetry Conventions](conventions.md) before preparing
  a new calculation.
- Use the repository calculations in [Examples](examples.md) as starting
  points.
- If the result is unexpected, consult
  [Troubleshooting and Compatibility](troubleshooting.md).
