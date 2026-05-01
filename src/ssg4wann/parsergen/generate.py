from collections import defaultdict
from datetime import datetime
import os
import numpy as np

def aveterms(entries_op, nsymm):

    acc = defaultdict(lambda: [0+0j, 0+0j])

    for d in entries_op:
        for key, value in d.items():
            s, n = acc[key]
            acc[key][0] = s + value[0]  # sum
    avg_dict = {k: total / nsymm for k, (total, count) in acc.items()}
    return avg_dict


def write(symmpath, entries_op, num_wann, nsymm): #not applied
    entsymm = aveterms(entries_op, nsymm)   
    reco = []
    for key, val in entsymm.items():
        R1, R2, R3, i, j = key
        reco.append(((int(R1), int(R2), int(R3), int(i), int(j)), complex(val)))
    R_set = { (R1, R2, R3) for (R1, R2, R3, i, j), H in reco }
    R_list = sorted(R_set)
    nrpts = len(R_list)
    Hmap = {key: H for key, H in reco}
    fulreco = []
    for R1, R2, R3 in R_list:
        for j in range(1, num_wann + 1):
            for i in range(1, num_wann + 1):

                key = (R1, R2, R3, i, j)

                H = Hmap.get(key, 0+0j)   # notexist = 0
                fulreco.append((key, H))

    fulreco.sort(key=lambda rec: (rec[0][0], rec[0][1], rec[0][2],
                                  rec[0][4], rec[0][3]))

    with open(symmpath, "w") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"# written at {now}\n")
        f.write(f"{num_wann:12d}\n")
        f.write(f"{nrpts:12d}\n")

        # figuring out
        degeneracies = [1] * nrpts
        for i in range(0, nrpts, 15):
            line = "".join(f"{d:5d}" for d in degeneracies[i:i+15])
            f.write(line + "\n")


        for (R1, R2, R3, i, j), H in fulreco:
            f.write(
                f"{R1:5d}{R2:5d}{R3:5d}{i:5d}{j:5d}"
                f"{H.real:22.16f}{H.imag:22.16f}\n"
            )

def outwrite(cwd, seed, reco, num_wann, nrpts, NONCOLLINEAR_channel):
    if NONCOLLINEAR_channel:
        reco.sort(key=lambda rec: (rec[0][0], rec[0][1], rec[0][2],
                                  rec[0][4], rec[0][3]))
        symmpath = os.path.join(cwd, seed + '_symmed_hr.dat')  

        with open(symmpath, "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"# written at {now}\n")
            f.write(f"{num_wann:12d}\n")
            f.write(f"{nrpts:12d}\n")

            # figuring out
            degeneracies = [1] * nrpts
            for i in range(0, nrpts, 15):
                line = "".join(f"{d:5d}" for d in degeneracies[i:i+15])
                f.write(line + "\n")


            for (R1, R2, R3, i, j), H in reco:
                f.write(
                    f"{R1:5d}{R2:5d}{R3:5d}{i:5d}{j:5d}"
                    f"{H.real:16.6f}{H.imag:16.6f}\n"
                )
    else:
        symmpath_up = os.path.join(cwd, seed + '.up_symmed_hr.dat')  
        symmpath_dn = os.path.join(cwd, seed + '.dn_symmed_hr.dat')  
        reco_up = []
        reco_dn = []
        for (R1, R2, R3, i, j), H in reco:
            if i <= num_wann and j <= num_wann:
                reco_up.append(((R1, R2, R3, i, j), H))
            else:
                reco_dn.append(((R1, R2, R3, i-num_wann, j-num_wann), H))
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

            # figuring out
            degeneracies = [1] * nrpts
            for i in range(0, nrpts, 15):
                line = "".join(f"{d:5d}" for d in degeneracies[i:i+15])
                f_up.write(line + "\n")
                f_dn.write(line + "\n")


            for (R1, R2, R3, i, j), H in reco_up:
                f_up.write(
                    f"{R1:5d}{R2:5d}{R3:5d}{i:5d}{j:5d}"
                    f"{H.real:16.6f}{H.imag:16.6f}\n"
                )
            for (R1, R2, R3, i, j), H in reco_dn:
                f_dn.write(
                    f"{R1:5d}{R2:5d}{R3:5d}{i:5d}{j:5d}"
                    f"{H.real:16.6f}{H.imag:16.6f}\n"
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
                        f.write(f"   {x:15.9f}   {e:16.6f}\n")
            f.write("\n")


    
def POSCAR_gen(lat, posi, INCAR_dir, spin_direction, NONCOLLINEAR_channel):
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
                raise ValueError(f"Warning: Unable to convert '{mag}' to float. Please check the MAGMOM values in the INCAR file.")

    else:
        final_magmom_values = expanded_magmoms

    with open('POSCAR', 'w') as f:
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

    return f"""# template input file 
# anything following '#', '!' or '//' in a line will be regard as comments
soc = {params['soc']}
SeedName='{params['seedname']}'  
use_win = '{params['use_win']}'
chnl = True
bands_trans = False
bands_num_points = 100
NONCOLLINEAR_channel = {params['noncollinear']}
spin_direction = 0 0 1
symm_output = True
"""