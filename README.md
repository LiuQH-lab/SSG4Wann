# SSG4Wann

A parallel (MPI-enabled) tool for **symmetrizing Wannier tight-binding Hamiltonians** (`*_hr.dat`) generated from Wannier90, using the Oriented Spin Space Group (SSG) symmetry of the magnetic system and supporting both strong and weak spin-orbit coupling limits.

---

## Table of Contents

- [SSG4Wann](#ssg4wann)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Repository Structure](#repository-structure)
  - [Requirements](#requirements)
  - [Rapid Example guide](#rapid-example-guide)
  - [Input Files](#input-files)
  - [Configuration (`sg.in`)](#configuration-sgin)
    - [Necessary keys](#necessary-keys)

    - [Optional keys](#optional-keys)

  - [Output Files](#output-files)
  - [Common errors:](#common-errors)
  - [Method Summary](#method-summary)
  - [License](#license)


---

## Overview

`SSG4Wann` is designed to restore or enforce symmetry constraints on Wannier Hamiltonians by averaging matrix elements under symmetry operations.  
It supports both collinear (up/down channels) and non-collinear workflows and is optimized for larger workloads through MPI-based parallel computation.

Typical workflow:

1. Read user configuration (`sg.in`)
2. Load Hamiltonian data (`*_hr.dat`)
3. Parse Wannier orbital/projection/lattice information
4. Construct symmetry action on orbital subspaces
5. Average transformed matrix elements over symmetry operations
6. Write symmetrized Hamiltonian output

---

## Key Features

-  Symmetrization of Wannier90 HR Hamiltonians with both MSG and SSG symmetries and support for spin channels and non-collinear settings
-  MPI parallelization for efficient processing of large Hamiltonians
- Configurable behavior through `sg.in` (e.g., channel mode, band transformation, k-point path)
- Optional band transformation workflow controls

---


## Requirements

- Python **3.12+** (3.14.3 recommended)
- `numpy`
- `pandas`
- `tqdm`
- `findspingroup` 


Optional but recommended:
- `mpi4py` 
- An MPI implementation:
  - OpenMPI, MPICH, Intel MPI, etc.

---

## Rapid Example guide


### 1)Installation

```bash
pip install ssg4wann
```

Python `>= 3.12` is required.
### 2) local minimum serial run example

ensure the following files are prepared in your working directory:
- hr file(s)
- win file(s)
- INCAR file 

#### a) python API example

```python
import ssg4wann as sw
sw.quick_run()
```
also you can set the working directory and the config file path in the `quick_run` function:

```pythonimport ssg4wann as sw
sw.quick_run(workdir="path/to/your/workdir", config_path="path/to/your/sg.in")
```
#### b) command line example

```bash
cd path/to/your/workdir
ssgsymm
```
or you can specify the config file path with the `-c` flag and the working directory with the `-w` flag:

```bash
ssgsymm -c config.in -w path/to/your/workdir
```

### 3) local parallel run example
install `mpi4py` and an MPI implementation (e.g., OpenMPI, MPICH, Intel MPI) in your local environment. Then you can run the code in parallel with `mpirun` command. 

```bash

mpirun --version
mpirun -np 4 ssgsymm -c config.in -w path/to/your/workdir
```

Note: When running in parallel in your local environment, you can control the number of processes with the -np flag. To prevent out-of-memory (OOM) issues, fewer processes may be safer for large structural systems.

### 4) parallel run on HPC cluster

In High-Performance Computing (HPC) clusters, the pre-built mpi4py wheel may conflict with the underlying MPI environment. It is strongly recommended to compile mpi4py from the source using the cluster's native MPI compiler.

#### Step 1: 
Ensure you have installed mpi4py in your environment which installed ssg4wann

#### Step 2:
Prepare a bash script (e.g., `job.lsf` or `job.sh`) in your work directory. Load your MPI module and run the ssg4wann, for example:
```bash
#!/bin/bash
# ... your job scheduler directives (e.g., #SBATCH) ...
module load mpi/2021.6.0
source /path/to/your/.venv/bin/activate
mpirun -np 56 ssgsymm -c config.in -w path/to/your/workdir
```




## Input Files

At minimum, prepare:

1. **Configuration**
    `sg.in` in your working directory

2. **Wannier Hamiltonian file(s)**  
	Depending on your channel mode:
    Non-collinear: `wannier90_hr.dat` 
    Collinear: `wannier90.up_hr.dat` and `wannier90.dn_hr.dat`

3. **Wannier metadata files** 

    Non-collinear: `wannier90.win` 
    Collinear: `wannier90.up.win` and `wannier90.dn.win`

    the code will read the necessary Wannier basis, lattice structure, projection information from the `.win` file(s).


5. **INCAR file** 
    The code will read `MAGMOM` in the INCAR file to determine the magnetic structure of the system, which is necessary for the correct symmetrization of the Hamiltonian.
    For collinear systems, the `MAGMOM` should be set to a single value per atom, while for non-collinear systems, the `MAGMOM` should be set to three values (x, y, z) per atom to specify the spin direction.

    If `ssg4wann` is going to generate the `sg.in` automatically, it will read `LNONCOLLINEAR` and `LSORBIT` tags in the INCAR file to determine the `soc` and `NONCOLLINEAR_channel` settings in the generated `sg.in`.

    

## Configuration (`sg.in`)

Example skeleton:

```ini
SeedName = 'wannier90'
mark = S
soc = False
use_win = wannier90.win
chnl = True
bands_trans = False
bands_num_points = 100
use_hr_file = 'wannier90_symmed_hr.dat'
NONCOLLINEAR_channel = true

```

### Necessary keys

#### SeedName tag
```ini
Tag name:   SeedName
Type:       String
Description:  base name for Wannier files (e.g., `wannier90`)
```



#### use_win tag
```ini
Tag name:   use_win
Type:       String (file path)
Description: path to Wannier90 `.win` file for orbital/projection/lattice info (e.g., `wannier90.win`)
```

#### NONCOLLINEAR_channel tag
```ini
Tag name:   NONCOLLINEAR_channel
Type:       Boolean (True/False)
Description: whether the system is in non-collinear channel. 
When `True`, the program will read the non-collinear HR file (`wannier90_hr.dat`) and perform symmetrization in the non-collinear channel. 
When `False`, the program will read the collinear HR files (`wannier90.up_hr.dat` and `wannier90.dn_hr.dat`) and perform symmetrization in the collinear channel. 
```

#### spin_direction tag
```ini
Tag name:   spin_direction
Type:       List of floats (e.g., '1 0 0')
Description: the spin quantization axis for symmetrization. 
It is  necessary to ensure that the spin direction is same as the `SAXIS` parameter in the VASP calculation when `NONCOLLINEAR_channel = True`. 
It is recommended to use the default `SAXIS = 0 0 1` for the VASP calculation. For the `NONCOLLINEAR_channel = True` case, this key is set to be `0 0 1` by default.
Only the correct setting of the spin direction can ensure the correct symmetrization of the Hamiltonian. 
This key is necessary when `NONCOLLINEAR_channel = False`.
```
#### soc tag
```ini
Tag name:   soc
Type:       Boolean (True/False)
Description: mark for spin-orbit coupling limit. 

When `False`, the program will perform the whole oriented spin space group to symmetrize the Hamiltonian. 

When `True`, the program will lower the symmetry to the corresponding subgroup of OSSG, which is equivalent to the magnetic space group (MSG) and perform the symmetrization with the MSG symmetry.
```



### Optional keys

#### chnl tag
```ini
Tag name:   chnl
Type:       Boolean (True/False)
Description: describes the spin sequencing for the Wannier basis. 
When `True`, the basis is ordered as [up1, up2, ..., upN, dn1, dn2, ..., dnN]. 
When `False`, the basis is ordered as [up1, dn1, up2, dn2, ..., upN, dnN]. It is set to `True` by default.
```
#### bands_trans tag
```ini
Tag name:   bands_trans
Type:       Boolean (True/False)
Description: whether to perform band structure transformation. When `True`, the program will read the specified HR file (see `use_hr_file` key) and calculate the band structure data. It is set to `False` by default.     
```

#### bands_num_points tag
```ini
Tag name:   bands_num_points
Type:       Integer
Description: number of k-points between each pair of k-points for band structure transformation. It is set to `100` by default. 
```
---

#### use_hr_file tag
```ini
Tag name:   use_hr_file
Type:       String (file path)
Description: path to the HR file for band structure transformation. This key is necessary when `bands_trans` is set to `True`.
```

#### begin kpoint_path ... end kpoint_path block
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

#### each_symm tag
```ini
Tag name:   each_symm
Type:       Boolean (True/False)
Description: Whether to output the symmetrized HR file for each symmetry operation. 
When `each_symm` is set to `True`, the program will output multiple HR files, which may cost more computational time. 
This tag is mainly for debugging and testing purposes and is set to `False` by default. 
```

#### hard_ave tag
```ini
Tag name:   hard_ave
Type:       Boolean (True/False)
Description: Whether to perform hard averaging of the transformed HR data. 
When `True`, the program will average the transformed HR data over all symmetry operations even though it does not contribute to the symmetrized entry. 
It may cost more computational time and output the symmetrized HR file with less accuracy. 
This tag is mainly for debugging and testing purposes and is set to `False` by default. 
```
#### symm_output tag
```ini
Tag name:   symm_output
Type:       String (file path)
Description: whether to output the group information of the given structure. 

The tag is set to be `False` by default.
```


## Output Files

Typical output includes symmetrized HR files or band structure data, depending on the configuration:

- `*_symmed_hr.dat`
- `*_bands.dat`

Output naming is controlled by seed/config and channel logic in the code.

---

## Common errors:
Even if you successfully run the code without any error, it is still possible that the symmetrization is not performed correctly due to incorrect input files or configuration.

- Your symmetrized band structure is completely different from the original one, but with the non-trivial dispersion and the same number of bands. This is most probably caused by incorrect setting of the spin direction. Ensure the spin direction is correctly specified!

- Your symmetrized band structure is partially the same as the original one, but with some **flat bands**, **extra bands**, **missing bands** or **connected bands**. This is most probably caused by the low quality of the wannierization process, even though the wannier band structure looks good. 
    - Ensure the wannierization disentanglement is well converged
    - Set `num_iter` as less as possible except you are certain that the you have constructed very perfect wannier functions. Check your `wannier90.wout` file to and ensure your center of wannier functions are not shifted too far from the original atomic positions.

## Method Summary

The core symmetrization pipeline is conceptually:

1. Build expressions of symmetry operators in orbital/spin basis
2. Map indices/orbitals under each operation
3. Find the lattice vectors set of the symmetrized Hamiltonian
4. Map the symmetrized entries to equivalent entries in the original HR data and average over contributing symmetry operations to get the symmetrized Hamiltonian.


## License
This project is licensed under the Apache License, Version 2.0.
See the LICENSE file for details.













