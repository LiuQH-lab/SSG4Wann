# Troubleshooting and Compatibility

Even if you successfully run the code without any error, it is still possible
that the symmetrization is not performed correctly due to incorrect input
files or configuration.

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

## Common errors

### The symmetrized band structure is completely different

Your symmetrized band structure is completely different from the original
one, but with the non-trivial dispersion and the same number of bands.

- Ensure the `spin direction` parameter is correctly specified.

- the wannier output Hamiltonian is under the basis of the wannier default
  squence, instead of the orbital squence you defined in the `win` file. Make
  sure that the `win` file which is inputed to `ssg4wann` has the same squence
  with the Hamiltonian file!!!

### The symmetrized band structure is partially the same

Your symmetrized band structure is partially the same as the original one,
but with some **flat bands**, **extra bands**, **missing bands** or **connected
bands**. This is most probably caused by the low quality of the wannierization
process, even though the wannier band structure looks good.

- Ensure the wannierization disentanglement is well converged
- Set `num_iter` as less as possible except you are certain that the you have
  constructed very perfect wannier functions. Check your `wannier90.wout` file
  to and ensure your center of wannier functions are not shifted too far from
  the original atomic positions.
- There might be some mistake with the wannierization process. Define atomic
  orbital for the initial guess as more as possible, even if they have higher
  eigenvalues than the energy window which is concerned. If the defined
  wannier basis is not enough, the disentanglement process may introduce the
  components of other orbitals which may cause the wrong result of
  symmetrization.

## VASP 6.2.0–6.3.0 non-collinear Wannier90 spin-axis bug { #vasp-noncollinear-spin-axis-bug }

VASP versions **6.2.0, 6.2.1, and 6.3.0** contain a known bug in the
Wannier90 interface for non-collinear/spinor calculations. The bug is fixed in
**VASP 6.3.1**. VASP 5.4.4 is not in the affected version range of this
specific issue. [VASP Known issues, issue #79](https://vasp.at/wiki/Known_issues)

Consequently, an intended Wannier90 spin quantization axis

```text
(qx, qy, qz)
```

is stored as

```text
(qz, qz, qz).
```

For the Wannier90 default quantization axis `(0, 0, 1)`, the effective axis
therefore becomes `(1, 1, 1)` after normalization.

For Hamiltonians generated using an affected VASP version with the default
Wannier90 spin quantization axis, use the following compatibility workaround
in `sg.in`:

```ini
spin_direction = 1 1 1
```

The recommended upstream solution is to regenerate the Wannier90 files using
VASP 5.4.4, 6.3.1 or newer.

The problem is only relevant for non-collinear calculations. A possible symptom of this bug is that the wrong band structure caused by this bug has the correct number of bands, correct degeneracies but the wrong dispersion. 


!!! note

    This issue concerns the spin quantization axis read by the Wannier90
    projection interface. It does not mean that VASP changed the global
    `SAXIS` value in the electronic-structure calculation.

