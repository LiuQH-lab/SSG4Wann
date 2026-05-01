from dataclasses import dataclass
import numpy as np
import re

@dataclass
class WannOrb:
    label: int
    L: int
    tau: np.ndarray
    global_index: int
    spin: str = None

def obnote(content):
    pattern = re.compile(r'begin projections(.*?)end projections', re.DOTALL)
    section = pattern.search(content)
    
    if section:
        projections = section.group(1).strip().splitlines()
        label = {}
        
        for line in projections:
            parts = line.split(":")
            if len(parts) == 2:
                atom, orbitals = parts
                # label += orbitals.strip().split(";")
                label[atom.strip()] = orbitals.strip().split(";")
        return label
    
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
                # coordinates = [float(parts[1 + i])/latti[i] for i in range(0,3)] 
                posi.append([element, coordinates])
        
        return posi
    
def angmap(label):
    if label == 's':
        L = 0
    elif label in ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']:
        L = 2
    elif label in ['px', 'py', 'pz']:
        L = 1
    else:   # f orb
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
        # ob = [label[at], np.float64(co[1])]
        wannobs.append(ob)
    k = 0
    for info in wannobs:    
        for i in info[0]:
            
            k += 1
            orbitals.append(WannOrb(i, L=angmap(i), tau= np.array(info[1]).reshape(3, 1), global_index = k))
    

    permutation, permuK = lat(content)  

        

            
    return orbitals, permutation, permuK, posi

