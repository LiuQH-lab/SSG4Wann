# Configuration Reference

## Configuration (`sg.in`)

`sg.in` is not necessarily required for running the code. If you do not
provide an `sg.in` file, the code will attempt to auto-detect the system
parameters and generate a it when running `ssg4wann`, and it will continue to
symmetrize the Hamiltonian with the auto-generated `sg.in`. Also you can use
`ssg4wann --init` command to generate the `sg.in` without running the
symmetrization.

Example skeleton:

```ini
SeedName = 'wannier90'
soc = False
use_win = wannier90.win
tb_mode = False
output_hr_from_tb = False
chnl = True
bands_trans = False
bands_num_points = 100
use_hr_file = 'wannier90_symmed_hr.dat'
use_tb_file = 'wannier90_symmed_tb.dat'
NONCOLLINEAR_channel = true
```

## Necessary keys

### SeedName tag

```ini
Tag name:   SeedName
Type:       String
Description:  base name for Wannier files (e.g., `wannier90`)
```

### use_win tag

```ini
Tag name:   use_win
Type:       String (file path)
Description: path to Wannier90 `.win` file for orbital/projection/lattice info (e.g., `wannier90.win`)
```

### NONCOLLINEAR_channel tag

```ini
Tag name:   NONCOLLINEAR_channel
Type:       Boolean (True/False)
Description: whether the system is in non-collinear channel.
When `True`, the program will read the non-collinear HR file (`wannier90_hr.dat`) and perform symmetrization in the non-collinear channel.
When `False`, the program will read the collinear HR files (`wannier90.up_hr.dat` and `wannier90.dn_hr.dat`) and perform symmetrization in the collinear channel.
```

### spin_direction tag

```ini
Tag name:   spin_direction
Type:       List of floats (e.g., '1 0 0')
Description: the spin quantization axis for symmetrization.

It is  necessary to ensure that the spin direction is same as the SAXIS parameter in the VASP calculation when `NONCOLLINEAR_channel = True`.
This key is necessary when `NONCOLLINEAR_channel = False`.
```

For more details, see [Spin direction](conventions.md#spin-direction).


### soc tag

```ini
Tag name:   soc
Type:       Boolean (True/False)
Description: mark for spin-orbit coupling limit.

When `False`, the program will perform the whole oriented spin space group to symmetrize the Hamiltonian.

When `True`, the program will lower the symmetry to the corresponding subgroup of OSSG, which is equivalent to the magnetic space group (MSG) and perform the symmetrization with the MSG symmetry.
```

## Optional keys

### tb_mode tag

```ini
Tag name:   tb_mode
Type:       Boolean (True/False)
Default:    False
Description: read Wannier90 `*_tb.dat` instead of `*_hr.dat`. The Hamiltonian
block uses the existing HR symmetrization, while the Cartesian position-matrix
block is symmetrized as a vector operator. The output is `*_symmed_tb.dat`.
```

### output_hr_from_tb tag

```ini
Tag name:   output_hr_from_tb
Type:       Boolean (True/False)
Default:    False
Description: when `tb_mode = True`, also write the symmetrized Hamiltonian
block in the standard Wannier90 HR format. The program reuses the Hamiltonian
already produced during TB symmetrization. It does not perform a second
symmetrization.

For a non-collinear calculation, the additional output is
`<SeedName>_symmed_hr.dat`. For a collinear calculation, the additional outputs
are `<SeedName>.up_symmed_hr.dat` and `<SeedName>.dn_symmed_hr.dat`.
This tag has no effect when `tb_mode = False`.
```

### chnl tag

```ini
Tag name:   chnl
Type:       Boolean (True/False)
Default:    True
Description: describes the spin sequencing for the Wannier basis. 
When `True`, the basis is ordered as [up1, up2, ..., upN, dn1, dn2, ..., dnN].
When `False`, the basis is ordered as [up1, dn1, up2, dn2, ..., upN, dnN]

When `NONCOLLINEAR_channel = False`, the `chnl` tag must be set to `True` for collinear calculations.
```

### bands_trans tag

```ini
Tag name:   bands_trans
Type:       Boolean (True/False)
Default:    False
Description: whether to perform band structure transformation. When `True`, the program will read the specified HR file (see `use_hr_file` key) and calculate the band structure data. It is set to `False` by default.
When `tb_mode = True` at the same time, the program instead reads the
Hamiltonian block from the TB file specified by `use_tb_file` and reuses the
same band calculation workflow. The position-matrix block in the TB file is
not used for the band calculation.
```

### bands_num_points tag

```ini
Tag name:   bands_num_points
Type:       Integer
Default:    100
Description: number of k-points between each pair of k-points for band structure transformation. This key is effective when `bands_trans` is set to `True`.
```

### use_hr_file tag

```ini
Tag name:   use_hr_file
Type:       String (file path)
Description: path to the HR file for band structure transformation. This key is necessary when `bands_trans` is set to `True`.
```

### use_tb_file tag

```ini
Tag name:   use_tb_file
Type:       String (file path)
Default:    wannier90_symmed_tb.dat
Description: path to the TB file for band structure transformation. This key
is used when both `bands_trans` and `tb_mode` are set to `True`. The
Hamiltonian block is read from this file and transformed with the same logic
used for an HR file.
```

### begin kpoint_path ... end kpoint_path block

```ini
Tag name:   begin kpoint_path ... end kpoint_path
Type:       Block of lines, each line containing a k-point label and its coordinates (e.g., `G 0.0 0.0 0.0`)
Description: defines the k-point path for band structure transformation. This block is necessary when `bands_trans` is set to `True`.
The k-point labels and coordinates should be specified in the same format as in wannier90 `.win` files. For example, you can specify:
begin kpoint_path
G 0.0 0.0 0.0 X 0.5 0.0 0.0
X 0.5 0.0 0.0 M 0.5 0.5 0.0
end kpoint_path
to define a k-point path from G to X to M.
```

### each_symm tag

```ini
Tag name:   each_symm
Type:       Boolean (True/False)
Description: Whether to output the symmetrized HR file for each symmetry operation.
When `each_symm` is set to `True`, the program will output multiple HR files, which may cost more computational time.
This tag is mainly for debugging and testing purposes and is set to `False` by default.
The `hard_ave` tag must be set to `True` when `each_symm` is set to `True`.
```

### hard_ave tag

```ini
Tag name:   hard_ave
Type:       Boolean (True/False)
Description: Whether to perform hard averaging of the transformed HR data.
When `True`, the program will average the transformed HR data over all symmetry operations even though it does not contribute to the symmetrized entry.
It may cost more computational time and output the symmetrized HR file with less accuracy.
This tag is mainly for debugging and testing purposes and is set to `False` by default.
```

### symm_output tag

```ini
Tag name:   symm_output
Type:       Boolean (True/False)
Description: whether to output the group information

The tag is set to be `True` by default.
```

### extend_LatVec tag

```ini
Tag name:   extend_LatVec
Type:       Boolean (True/False)
Description: whether to extend the lattice vectors set of the symmetrized Hamiltonian.
When `True`, the program will extend the lattice vectors set which is generated by the operators.
When `False`, the program will use the lattice vectors set which is same as the original hr file
this tag.

The tag is set to be `True` by default.
```

### forced_hermitianize

```ini
Tag name:   forced_hermitianize
Type:       Boolean(True/False)
Description: whether to output the Hamiltonian with the hermitian forcing process.
The output Hamiltonian is hermitian if an hermitian is inputed. Anyway if the original Hamiltonian is not hermitian strictly, you can set this tag to `True` for test. It will output the symmetrized Hamiltonian by averaging the Hamiltonian with its hermitian conjugate. 

The tag is set to be `False` by default.
```

### spinonly_speedup

```ini
Tag name:   spinonly_speedup
Type:       Boolean(True/False)
Description: whether to perform the direct product structure to speed up the symmetrization.
Default:    True

When `True`, the program will only perform the NONTRIVIAL SPIN GROUP operation when symmetrization without SOC, and detect the spin-only group to ensure whether the Hamiltonian is real.
When `False`, the program will perform the whole oriented spin space group operation when calculating without SOC, which is more time-consuming.
This tag is mainly for testing purposes.

It is intentionally not applied in `tb_mode`, because the position-matrix
block must currently be averaged over the full oriented spin space group.
```