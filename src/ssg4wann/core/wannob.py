from dataclasses import dataclass
from pathlib import Path
import numpy as np
import re

DEFAULT_ORBITALS = {
    1: ['pz', 'px', 'py'],
    2: ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'],
    3: ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)']
}

@dataclass
class WannOrb:
    label: int
    L: int
    tau: np.ndarray
    global_index: int
    spin: str = None

def obnote(content):
    pattern = re.compile(r'(?i)begin\s+projections(.*?)end\s+projections', re.DOTALL)
    section = pattern.search(content)
    
    if section:
        projections = section.group(1).strip().splitlines()
        label = {}
        
        shell_expansion = {
            's': ['s'],
            'p': DEFAULT_ORBITALS[1],
            'd': DEFAULT_ORBITALS[2],
            'f': DEFAULT_ORBITALS[3]
        }
        
        for line in projections:
            line = line.split('#')[0].split('!')[0].strip()
            if not line or ':' not in line:
                continue
                
            parts = line.split(":")
            if len(parts) == 2:
                atom = parts[0].strip()
                orbitals_raw = re.split(r'[,;]', parts[1].strip().replace(" ", ""))
                
                expanded_orbitals = []
                for orb in orbitals_raw:
                    if not orb:
                        continue
                    

                    if orb.lower() in shell_expansion:
                        expanded_orbitals.extend(shell_expansion[orb.lower()])
                    else:
                        expanded_orbitals.append(orb)
                        
                label[atom] = expanded_orbitals
        return label
    
    else:
        raise ValueError("No valid 'begin projections ... end projections' block found in the .win file. Please check your .win file format and ensure it contains a properly formatted projections section.")
def lat(content):
    pattern = re.compile(r'begin unit_cell_cart(.*?)end unit_cell_cart', re.DOTALL)
    section = pattern.search(content)
    
    if section:
        lattice_lines = section.group(1).strip().splitlines()
        a1, a2, a3 = map(float, lattice_lines[0].split())
        b1, b2, b3 = map(float, lattice_lines[1].split())
        c1, c2, c3 = map(float, lattice_lines[2].split())
        omega = (a1*(b2*c3 - b3*c2) - a2*(b1*c3 - b3*c1) + a3*(b1*c2 - b2*c1)) 

        e1, e2, e3 =  (b2*c3 - b3*c2)/omega, (b3*c1 - b1*c3)/omega, (b1*c2 - b2*c1)/omega
        f1, f2, f3 =  (c2*a3 - c3*a2)/omega, (c3*a1 - c1*a3)/omega, (c1*a2 - c2*a1)/omega
        g1, g2, g3 =  (a2*b3 - a3*b2)/omega, (a3*b1 - a1*b3)/omega, (a1*b2 - a2*b1)/omega

        permutation = np.array([[a1, b1, c1],
                                [a2, b2, c2],
                                [a3, b3, c3]])
        
        permuK = 2 * np.pi * np.array([[e1, e2, e3],
                                       [f1, f2, f3],
                                       [g1, g2, g3]])
        return permutation, permuK
    
    
def coordi(content):
    pattern = re.compile(r'begin atoms_cart(.*?)end atoms_cart', re.DOTALL)
    section = pattern.search(content)
    permutation, permuK = lat(content)
    
    if section:
        atoms = section.group(1).strip().splitlines()
        posi = []
        
        for line in atoms:
            parts = line.split()
            if len(parts) == 4:
                element = parts[0]
                cart_coe = np.array([float(parts[1]), float(parts[2]), float(parts[3])]).reshape(3, 1)
                latt_coe = np.linalg.inv(permutation) @ cart_coe
                coordinates = [latt_coe[i][0] for i in range(0,3)] 
                posi.append([element, coordinates])
        
        return posi
    
def angmap(label):
    if label == 's':
        L = 0
    elif label in ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']:
        L = 2
    elif label in ['px', 'py', 'pz']:
        L = 1
    else:   
        L = 3
    return L

def wannobs(winpath):
    """define the wannier basis with its information and find the permutation matrices"""
    with open(winpath,'r') as f:
        content = f.read()
    orbitals = []
    label = obnote(content)
    posi = coordi(content)
    wannobs = []
    for co in posi:
        at = co[0]
        ob = [label.get(at, []), np.float64(co[1])]
        wannobs.append(ob)
    k = 0
    for info in wannobs:    
        for i in info[0]:
            
            k += 1
            orbitals.append(WannOrb(i, L=angmap(i), tau= np.array(info[1]).reshape(3, 1), global_index = k))
    

    permutation, permuK = lat(content)  

        
    return orbitals, permutation, permuK, posi





def proj_seq(win_path: str) -> dict:
    win_path = Path(win_path)
    if win_path.exists():

        win_content = win_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"win_path {win_path} does not exist or is not a file.")

    match = re.search(r"(?i)begin\s+projections\s*(.*?)\s*end\s+projections", win_content, re.DOTALL)


        
    proj_text = match.group(1)
    custom_orders = {}
    

    orb_to_L = {o: L for L, orbs in DEFAULT_ORBITALS.items() for o in orbs}

    for line in proj_text.split('\n'):

        line = line.split('#')[0].split('!')[0].strip()
        if not line:
            continue
            

        if ':' not in line:
            continue
            
        orbs_str = line.split(':')[1].strip().replace(" ", "")
        orbs_list = [o for o in re.split(r'[,;]', orbs_str) if o]
        
        current_line_orders = {1: [], 2: [], 3: []}
        for o in orbs_list:
            if o in orb_to_L:
                current_line_orders[orb_to_L[o]].append(o)
                
        for L, orbs in current_line_orders.items():
            if not orbs:
                continue 
            
            if L in custom_orders:
                if custom_orders[L] != orbs:
                    raise ValueError(
                        f"conflicting orbital order for L={L}: {custom_orders[L]} vs {orbs}, check your .win file's projection section for consistency!" 
                    )
            else:
                custom_orders[L] = orbs
    if custom_orders == {}:
        print(f"Warning: no detailed projection information found in {win_path}! the projection sequence will be set to default order")
    custom_orders = DEFAULT_ORBITALS | custom_orders
    return custom_orders