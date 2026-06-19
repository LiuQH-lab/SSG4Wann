# Examples

The repository provides four example calculations covering collinear,
non-collinear, HR, and TB workflows. The examples contain input files and
reference outputs that can be used to check an installation or as starting
points for a new calculation.

## Before running an example

Install SSG4Wann and confirm that the command-line interface is available:

```bash
python -m pip install ssg4wann
ssg4wann --version
```

Run examples in a copied directory so that the reference output files in the
repository are not overwritten:

```bash
cp -R examples/Fe run-Fe
cd run-Fe
ssg4wann -c sg.in
```

When SSG4Wann starts, it reads `sg.in` from the working directory. You can also
run an example from another directory:

```bash
ssg4wann -c sg.in -w /path/to/run-Fe
```

For a collinear calculation, verify `spin_direction` before running. It must
describe the spin quantization axis used by the first-principles calculation.
For a non-collinear calculation, it must be consistent with the VASP `SAXIS`
convention.

## Available examples

| Directory | Input mode | Spin treatment | Symmetry used | Main output |
| --- | --- | --- | --- | --- |
| `examples/Fe` | HR | Collinear, no SOC | OSSG | `wannier90.up_symmed_hr.dat`, `wannier90.dn_symmed_hr.dat` |
| `examples/Fe_SOC` | HR | Non-collinear, SOC | MSG | `wannier90_symmed_hr.dat` |
| `examples/Fe_tb` | TB | Collinear, no SOC | OSSG | `wannier90.up_symmed_tb.dat`, `wannier90.dn_symmed_tb.dat` |
| `examples/Nb3VS6` | HR | Collinear, no SOC | OSSG | `wannier90.up_symmed_hr.dat`, `wannier90.dn_symmed_hr.dat` |

In SSG4Wann, the `soc` setting selects the symmetry operation set:

- `soc = False` uses the full oriented spin space group (OSSG).
- `soc = True` uses the corresponding magnetic space group (MSG) subgroup.

## Fe: collinear HR symmetrization

This is the smallest collinear example. It reads separate spin-up and
spin-down Wannier90 Hamiltonians:

```text
wannier90.up_hr.dat
wannier90.dn_hr.dat
```

The important settings are:

```ini
soc = False
SeedName = 'wannier90'
use_win = 'wannier90.up.win'
NONCOLLINEAR_channel = False
chnl = True
spin_direction = 1.524205 -1.077775 1.866762
```

Run it from a working copy:

```bash
cp -R examples/Fe run-Fe
cd run-Fe
ssg4wann -c sg.in
```

The main generated files are:

```text
wannier90.up_symmed_hr.dat
wannier90.dn_symmed_hr.dat
ssg_symm.json
```

`ssg_symm.json` contains the symmetry information returned by
`findspingroup`.

## Fe_SOC: non-collinear HR symmetrization

This example demonstrates a spinor Hamiltonian with spin-orbit coupling. It
uses a single Wannier90 HR file:

```text
wannier90_hr.dat
```

The important settings are:

```ini
soc = True
SeedName = 'wannier90'
use_win = 'wannier90.win'
NONCOLLINEAR_channel = True
spin_direction = 0 0 1
```

Run it with:

```bash
cp -R examples/Fe_SOC run-Fe_SOC
cd run-Fe_SOC
ssg4wann -c sg.in
```

The main output is:

```text
wannier90_symmed_hr.dat
```

For non-collinear calculations, `spin_direction` must use the same spin
coordinate convention as the original calculation. With the usual VASP
default `SAXIS`, use `spin_direction = 0 0 1`.

## Fe_tb: TB Hamiltonian and position matrix

This example reads Wannier90 `*_tb.dat` files. In addition to the Hamiltonian,
SSG4Wann symmetrizes the three Cartesian components of the position matrix as
a vector operator.

The important settings are:

```ini
soc = False
tb_mode = True
output_hr_from_tb = False
NONCOLLINEAR_channel = False
```

Run it with:

```bash
cp -R examples/Fe_tb run-Fe_tb
cd run-Fe_tb
ssg4wann -c sg.in
```

The main outputs are:

```text
wannier90.up_symmed_tb.dat
wannier90.dn_symmed_tb.dat
```

To also export the symmetrized Hamiltonian blocks in standard HR format, set:

```ini
output_hr_from_tb = True
```

This additionally writes:

```text
wannier90.up_symmed_hr.dat
wannier90.dn_symmed_hr.dat
```

## Nb3VS6: multi-atom collinear example

This example demonstrates a larger collinear magnetic system. It is useful for
checking orbital mapping, atom mapping, and the treatment of a nontrivial
magnetic structure.

The important settings are:

```ini
soc = False
NONCOLLINEAR_channel = False
spin_direction = 1 0 0
```

Run it with:

```bash
cp -R examples/Nb3VS6 run-Nb3VS6
cd run-Nb3VS6
ssg4wann -c sg.in
```

This calculation is larger than the Fe examples and is a better candidate for
an MPI run:

```bash
mpirun -np 4 ssg4wann -c sg.in
```

## Checking a result

A successful run should produce the expected `*_symmed_hr.dat` or
`*_symmed_tb.dat` files without basis-mapping or spin-rotation errors.
Successful completion alone does not guarantee a physically correct result.
Before using the model, check at least the following:

1. Compare the original and symmetrized band structures.
2. Check that the chosen `spin_direction` matches the original calculation.
3. Check that the projection order in the `.win` file matches the basis order
   of the HR or TB file.
4. Inspect warnings concerning Wannier centers, orbital matching, or symmetry
   operations.
5. Confirm that the symmetrized model has the intended OSSG or MSG symmetry.

Large changes in dispersion, unexpected flat bands, missing bands, or extra
band crossings commonly indicate an inconsistent basis order, an incorrect
spin direction, or insufficient Wannierization quality.
