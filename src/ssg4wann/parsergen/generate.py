from collections import defaultdict
from datetime import datetime
import os
import numpy as np

from ..exceptions import ConfigParseError, WannierMatchError

def aveterms(entries_op, nsymm):

    acc = defaultdict(lambda: [0+0j, 0+0j])

    for d in entries_op:
        for key, value in d.items():
            s, n = acc[key]
            acc[key][0] = s + value[0]  # sum
    avg_dict = {k: total / nsymm for k, (total, count) in acc.items()}
    return avg_dict

def _flatten_operation_result(raw_op_data):
    flat_reco = []
    for element in raw_op_data:
        if isinstance(element, list) and (
            len(element) != 2 or not isinstance(element[0], tuple)
        ):
            flat_reco.extend(element)
        else:
            flat_reco.append(element)
    return flat_reco


def _average_operation_results(results, nsymm):
    per_operation = {}
    terms = []
    for fallback_idx, item in enumerate(results):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
            idx, raw_op_data = item
        else:
            idx, raw_op_data = fallback_idx, item
        flat_reco = _flatten_operation_result(raw_op_data)
        per_operation[idx] = flat_reco
        terms.extend({coords: [value]} for coords, value in flat_reco)

    averaged = aveterms(terms, nsymm)
    return [[coords, value] for coords, value in averaged.items()], per_operation



def outwrite(cwd, seed, reco, num_wann, nrpts, NONCOLLINEAR_channel, chnl):
    if NONCOLLINEAR_channel:
        reco.sort(key=lambda rec: (rec[0][0], rec[0][1], rec[0][2],
                                  rec[0][4], rec[0][3]))
        symmpath = os.path.join(cwd, seed + '_symmed_hr.dat')  

        with open(symmpath, "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"# written at {now}\n")
            f.write(f"{num_wann:12d}\n")
            f.write(f"{nrpts:12d}\n")

            degeneracies = [1] * nrpts
            for i in range(0, nrpts, 15):
                line = "".join(f"{d:5d}" for d in degeneracies[i:i+15])
                f.write(line + "\n")


            for (R1, R2, R3, i, j), H in reco:
                f.write(
                    f"{R1:5d}{R2:5d}{R3:5d}{i:5d}{j:5d}"
                    f"{H.real:22.16f}{H.imag:22.16f}\n"
                )
    else:
        symmpath_up = os.path.join(cwd, seed + '.up_symmed_hr.dat')  
        symmpath_dn = os.path.join(cwd, seed + '.dn_symmed_hr.dat')  
        reco_up = []
        reco_dn = []
        for (R1, R2, R3, i, j), H in reco:
            match chnl:
                case True:         # upupdndn format
                    if i <= num_wann and j <= num_wann:
                        reco_up.append(((R1, R2, R3, i, j), H))
                    elif i > num_wann and j > num_wann:
                        reco_dn.append(((R1, R2, R3, i-num_wann, j-num_wann), H))
                    else:
                        raise WannierMatchError(f"Error: Inconsistent channel format. Found i={i}, j={j} with num_wann={num_wann}. Please check the chnl setting and the Hamiltonian file.")
                case False:      # updnupdn format
                    if i % 2 == 1 and j % 2 == 1:
                        reco_up.append(((R1, R2, R3, i, j, H)))
                    elif i % 2 == 0 and j % 2 == 0:
                        reco_dn.append(((R1, R2, R3, i-1, j-1), H))
                    else:
                        raise WannierMatchError(f"Error: Inconsistent channel format. Found i={i}, j={j} with chnl={chnl}. Please check the chnl setting and the Hamiltonian file.")
        reco_up.sort(key=lambda rec: (rec[0][0], rec[0][1], rec[0][2],
                                  rec[0][4], rec[0][3]))
        reco_dn.sort(key=lambda rec: (rec[0][0], rec[0][1], rec[0][2],
                                  rec[0][4], rec[0][3]))
        with open(symmpath_up, "w") as f_up, open(symmpath_dn, "w") as f_dn:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f_up.write(f"# written at {now}\n")
            f_up.write(f"{num_wann:12d}\n")
            f_up.write(f"{nrpts:12d}\n")

            f_dn.write(f"# written at {now}\n")
            f_dn.write(f"{num_wann:12d}\n")
            f_dn.write(f"{nrpts:12d}\n")

            degeneracies = [1] * nrpts
            for i in range(0, nrpts, 15):
                line = "".join(f"{d:5d}" for d in degeneracies[i:i+15])
                f_up.write(line + "\n")
                f_dn.write(line + "\n")


            for (R1, R2, R3, i, j), H in reco_up:
                f_up.write(
                    f"{R1:5d}{R2:5d}{R3:5d}{i:5d}{j:5d}"
                    f"{H.real:22.16f}{H.imag:22.16f}\n"
                )
            for (R1, R2, R3, i, j), H in reco_dn:
                f_dn.write(
                    f"{R1:5d}{R2:5d}{R3:5d}{i:5d}{j:5d}"
                    f"{H.real:22.16f}{H.imag:22.16f}\n"
                )

def bandwrite(bandspath, x_axis, eigenvalues, hr4trans, labels):
    num_bands = eigenvalues.shape[1]
    num_k = len(x_axis)
    with open(bandspath, 'w') as f:
        f.write(f"# Bands derived from {hr4trans}\n")
        f.write(f"#      k-path len        Energy\n")
        for b_idx in range(num_bands):
            for k_idx in range(num_k):
                x = x_axis[k_idx]
                e = eigenvalues[k_idx, b_idx]
                f.write(f"   {x:15.9f}   {e:16.6f}\n")
                if x in dict(labels).keys() and x != 0 :
                    f.write("\n")  
                    if x != x_axis[-1]:
                        f.write(f"   {x:15.9f}   {e:25.15f}\n")
            f.write("\n")


    
def POSCAR_gen(lat, posi, INCAR_dir, spin_direction, NONCOLLINEAR_channel, workdir):
    magmom_str_lines = ""
    is_reading_magmom = False
    

    try:
        magmom_str_lines = ""
        is_reading_magmom = False

        with open(INCAR_dir, 'r') as f:
            for line in f:

                clean_line = line.split('#')[0].split('!')[0].strip()
                

                if not clean_line:
                    continue
                if 'MAGMOM' in clean_line and not is_reading_magmom:
                    is_reading_magmom = True

                    content = clean_line.split('=', 1)[1].split(';')[0].strip()
                    
                    if content.endswith('\\'):
                        magmom_str_lines += " " + content[:-1].strip()
                    else:
                        magmom_str_lines += " " + content
                        break  
                        
                elif is_reading_magmom:

                    if '=' in clean_line:
                        break  
                        

                    content = clean_line.split(';')[0].strip()
                    if content.endswith('\\'):
                        magmom_str_lines += " " + content[:-1].strip()
                    else:
                        magmom_str_lines += " " + content
                        break  
    except FileNotFoundError:
        raise FileNotFoundError(f"INCAR file not found at the directory: {INCAR_dir}")


    raw_magmoms = magmom_str_lines.split()
    expanded_magmoms = []
    for val in raw_magmoms:
        if '*' in val:
            count, mag_val = val.split('*')
            expanded_magmoms.extend([mag_val] * int(count))
        else:
            expanded_magmoms.append(val)

    final_magmom_values = []
    if not NONCOLLINEAR_channel:
        unit = spin_direction / np.linalg.norm(spin_direction)
        for mag in expanded_magmoms:
            try:
                mag_float = float(mag)
                mag_vector = mag_float * unit

                final_magmom_values.append(f"{mag_vector[0]:.6f} {mag_vector[1]:.6f} {mag_vector[2]:.6f}")
            except ValueError:
                raise ConfigParseError(f"Warning: Unable to convert '{mag}' to float. Please check the MAGMOM values in the INCAR file.")

    else:
        final_magmom_values = expanded_magmoms

    with open(os.path.join(workdir, 'POSCAR'), 'w') as f:
        f.write('Generated POSCAR from win with the magnetic moments\n')
        f.write('1.0\n')
        lat = lat.T
        for vec in lat:
            f.write(f"{vec[0]:16.9f} {vec[1]:16.9f} {vec[2]:16.9f}\n")
        
        elements = [co[0] for co in posi]
        
        element_counts = {}
        for el in elements:
            if el not in element_counts:
                element_counts[el] = 0
            element_counts[el] += 1
            
        for el in element_counts.keys():
            f.write(f"{el} ")
        f.write("\n")
        
        for count in element_counts.values():
            f.write(f"{count} ")
        f.write("\n")
        
        f.write("direct\n")
        for co in posi:
            cart_coe = co[1]
            f.write(f"{cart_coe[0]:16.9f} {cart_coe[1]:16.9f} {cart_coe[2]:16.9f}\n")

        magmom_str = " ".join(final_magmom_values)
        f.write(f"# MAGMOM = {magmom_str}\n")


def get_sg_template(params: dict) -> str:
    template = f"""# input file for the SSG4Wann package, generated automatically at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# '#', '!' or '//' will be regard as comments

# SOC flag True for OSSG or False for subgroup MSG
soc = {params['soc']}

# seed name of the Hamiltonian file
SeedName='{params['seedname']}'  

# .win file
use_win = '{params['use_win']}'

# read and symmetrize *_tb.dat instead of *_hr.dat
tb_mode = {params.get('tb_mode', 'False')}

# also write the symmetrized Hamiltonian block in HR format in tb mode
output_hr_from_tb = False

# spin channel format: 'updnupdn' (False) or 'upup...dndn...' (True)
chnl = True

# transform hr file to band structure data
bands_trans = False

# the hr file transformed to the band structure
use_hr_file = wannier90_symmed_hr.dat

# number of k-points for band between each 2 high-symmetry k points
bands_num_points = 100

# NONCOLINEAR flag True or False or T or F
NONCOLLINEAR_channel = {params['noncollinear']}

# direction of the spin polarization
spin_direction = 0 0 1

# output the OSSG information
symm_output = True
"""
    
    if params.get('kpoint_path'):
        template += "\nbegin kpoint_path\n"
        template += params['kpoint_path']
        template += "\nend kpoint_path\n"
        
    return template


def _write_hr_from_tb(config, workdir, Hsymm, num_wann, nrpts):
    if not (config.tb_mode and config.output_hr_from_tb):
        return

    print("Writing symmetrized Hamiltonian block in HR format...")
    outwrite(
        workdir,
        config.seed,
        reco=Hsymm,
        num_wann=num_wann,
        nrpts=nrpts,
        NONCOLLINEAR_channel=config.NONCOLLINEAR_channel,
        chnl=config.chnl,
    )
