# Basis, Spin and Symmetry Conventions

The spin direction, Wannier projection order, atomic positions, and
Hamiltonian basis must be mutually consistent. This page collects the
conventions that are required when preparing an SSG4Wann calculation.

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

## Spin direction

For a collinear calculation, verify `spin_direction` before running. It specifies the common
spin-alignment direction in Cartesian coordinates, not in direct
lattice coordinates. Each scalar MAGMOM value m_i read from INCAR is converted
to the vector

m_i * spin_direction / |spin_direction|

and passed to `findspingroup` through the embedded MAGMOM entry in the generated
POSCAR. The magnitude of spin_direction is ignored; only its direction is used.
 The included [example](examples.md#Fe) primitive bcc Fe cell with a collinear spin should use `spin_direction = 1.524205 -1.077775 1.866762`. 


For non-collinear calculations, `spin_direction` must use the same spin
coordinate convention as the original calculation. With the usual VASP
default `SAXIS`, use `spin_direction = 0 0 1`.

For the VASP
6.2.0–6.3.0 exception to the usual default-axis rule, see
[Troubleshooting and Compatibility](troubleshooting.md#vasp-noncollinear-spin-axis-bug).

## Wannier basis order

The `.win` file provides the projection order used to interpret the
Hamiltonian basis. Check that the projection order in the `.win` file matches
the basis order of the HR or TB file.

The `chnl` tag describes the spin sequencing for the Wannier basis. See the
[chnl tag](configuration.md#chnl-tag) for the two supported orderings.

## Method Summary

The core symmetrization pipeline is conceptually:

1. Build expressions of symmetry operators in orbital/spin basis
2. Map indices/orbitals under each operation
3. Find the lattice vectors set of the symmetrized Hamiltonian
4. Map the symmetrized entries to equivalent entries in the original HR data
   and average over contributing symmetry operations to get the symmetrized
   Hamiltonian.