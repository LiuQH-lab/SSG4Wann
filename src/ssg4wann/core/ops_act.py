import numpy as np
from .cartesian_tensors import rotation_to_cubic_dmatrix
from dataclasses import InitVar, dataclass

from .constants import sigma_x, sigma_y, sigma_z, TOL_SPIN_DET, TOL_ROTATION_AXIS, TOL_MATRIX_ZERO, PI, DEFAULT_ORBITALS
from ..exceptions import SpinRotationError, WannierMatchError

type ColVec = np.ndarray 
type Matrix3x3 = np.ndarray
type Spinor = np.ndarray

def permuspinget(spin_direction: ColVec) -> Matrix3x3:
    spin_direction = spin_direction / np.linalg.norm(spin_direction)
    rotaxis = np.cross([0, 0, 1], spin_direction)
    if np.linalg.norm(rotaxis) < TOL_ROTATION_AXIS:
        return np.eye(3)
    else:
        rotaxis = rotaxis / np.linalg.norm(rotaxis)
        theta = np.arccos(np.dot([0, 0, 1], spin_direction) / np.linalg.norm(spin_direction))
        K = np.array([[0, -rotaxis[2], rotaxis[1]],
                        [rotaxis[2], 0, -rotaxis[0]],
                        [-rotaxis[1], rotaxis[0], 0]])
        permuspin = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        permuspin[np.abs(permuspin) < TOL_SPIN_DET] = 0.0
        if theta > PI or theta < TOL_MATRIX_ZERO:
            raise SpinRotationError(f"Error in calculating permuspin matrix for spin direction {spin_direction}. Check your input symmetry operation or structure file!!!")
        if abs(np.linalg.det(permuspin) - 1.0) > TOL_SPIN_DET:
            raise SpinRotationError(f"Error in calculating permuspin matrix for spin direction {spin_direction}. The determinant of the permuspin matrix must be 1.0, but got {np.linalg.det(permuspin)}. Check your input symmetry operation or structure file!!!")
        return permuspin
        
def rotget(rot: Matrix3x3, spin_direction: ColVec) -> Matrix3x3:
    permuspin = permuspinget(spin_direction)
    rot = permuspin.T @ rot @ permuspin
    det = np.linalg.det(rot)
    positive = rot if det > 0 else - rot
    vals, vecs = np.linalg.eig(positive)
    idx = np.argmin(np.abs(vals - 1.0))
    axis = vecs[:, idx].reshape(3, -1) 
    tr = np.trace(positive)
    cos_theta = np.clip((tr - 1) / 2.0, -1.0, 1.0)
    v_antisym = np.array([
        positive[2, 1] - positive[1, 2],
        positive[0, 2] - positive[2, 0],
        positive[1, 0] - positive[0, 1]
    ])
    sin_theta = (np.dot(v_antisym, axis) / 2.0).real
    
    if abs(sin_theta) < TOL_MATRIX_ZERO and abs(cos_theta) - 1.0 < TOL_MATRIX_ZERO:
        theta = np.arccos(cos_theta)
    else:
        theta = np.arctan2(sin_theta, cos_theta)
    term_sigma = axis[0]*sigma_x + axis[1]*sigma_y + axis[2]*sigma_z
    U = np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * term_sigma
    if np.linalg.det(U) - 1.0 > TOL_SPIN_DET:
        raise SpinRotationError("Error in calculating U matrix for spin rotation. Check your input symmetry operation or structure file!!!")
        
    return U



@dataclass(slots=True)
class ops_actclass:
    matrix: np.ndarray
    translation: np.ndarray
    permutation: InitVar[np.ndarray]
    spin_direction: np.ndarray
    
    def __post_init__(self, permutation):
        self.rot_cart = permutation @ self.matrix @ np.linalg.inv(permutation)
        self.lattice_cart = permutation
        self.translation_cart = permutation @ self.translation
        
    @property
    def is_time_reversed(self) -> bool:
        
        raise NotImplementedError("the inheritance class must implement the time reversal property, which is crucial for determining the correct spin rotation behavior in the i_find method. Please implement this property in the Mops and Sops classes accordingly.")
    

    def i_find(self, i, repdict, orbSpin) -> list[tuple[int, complex]]:
        from .map import revmapsp, formapsp

        label, L, tau, spin = formapsp(i, orbSpin)
        tauNew = self.tau_find(tau)
        LNew = L
        
        spinor = np.array([[1],[0]], dtype=complex) if spin == "up" else np.array([[0],[1]], dtype=complex)


        if not self.is_time_reversed:
            spinorNew = self.U @ spinor
        else:
            spinorNew = self.U @ np.array([[0, -1],[1, 0]]) @ spinor
            
        hopnew = []
        
        for n, SpindNew in enumerate(spinorNew[:, 0]):
            if abs(SpindNew) > TOL_MATRIX_ZERO:
                spinNew = "up" if n == 0 else "dn"
                basvec = np.zeros((2*L+1, 1))
                
                if L == 0:
                    j = 1
          
                elif L in DEFAULT_ORBITALS:
                    if label not in DEFAULT_ORBITALS[L]:
                        raise WannierMatchError(f"Error in calculating the new orbital index for i={i}, label={label}, L={L}. The label is not found in the default orbital sequence for L={L}. Check your input symmetry operation or wannier90.win file!!!")
                    j = DEFAULT_ORBITALS[L].index(label) + 1
                else:
                    raise WannierMatchError(f"Unsupported angular momentum L={L}")
                basvec[j-1] = 1.0
                try:
                    basvecNew = repdict[L] @ basvec
                except ValueError:
                    raise WannierMatchError(f"matrix for L={L} is {repdict.get(L)}, base vec = {basvec}. ")
                Angind = np.where(np.abs(basvecNew) > TOL_MATRIX_ZERO)[0] + 1
                
                for AngindNew in Angind:
                    if L == 0:
                        labelNew = label
                
                    elif L in DEFAULT_ORBITALS:
                        labelNew = DEFAULT_ORBITALS[L][AngindNew - 1]
                    else:
                        raise WannierMatchError(f"Unsupported angular momentum L={L}")
                    coe = SpindNew * (basvecNew[AngindNew - 1, 0])
                   
                    if (inew := revmapsp(labelNew, LNew, tauNew, spinNew, orbSpin)) is  None:
                        raise WannierMatchError(f"Error in calculating the new orbital index for i={i}, labelNew={labelNew}, LNew={LNew}, tauNew={tauNew}, spinNew={spinNew}. The new orbital index is not found in the wannier orbital list. Check your input symmetry operation or wannier90.win file!!!")
                    hopnew.append((inew, coe))

        if (abs(norm := sum(np.abs(coe)**2 for _, coe in hopnew))-1) > TOL_MATRIX_ZERO:
            raise WannierMatchError(f"Error in calculating the new orbital index and coefficient for i={i}, operator = {self}. The norm of the coefficients should not be greater than 1, but got {norm}. Check your input symmetry operation or wannier90.win file!!!")
        return hopnew
    
    def tau_find(self, tau:ColVec) -> ColVec:
        tauNew = self.matrix @ tau + self.translation
        return tauNew - np.floor(tauNew)
   

    def rep_find(self) -> dict:
        prep, drep, frep = rotation_to_cubic_dmatrix(self.rot_cart, L=1), rotation_to_cubic_dmatrix(self.rot_cart, L=2), rotation_to_cubic_dmatrix(self.rot_cart, L=3)
        repdict = {0: np.array([[1]]) , 1: np.array(prep.tolist(), dtype=float), 2: np.array(drep.tolist(), dtype=float), 3: np.array(frep.tolist(), dtype=float)}
        return repdict
    
    def R_find(self, i, j, R, orbitals):
        from .map import formapsp
        _, _, taui, _ = formapsp(i, orbitals)
        _, _, tauj, _ = formapsp(j, orbitals)
        Rvec = np.array(R).reshape(3, 1)

        if np.linalg.norm(taui) < TOL_MATRIX_ZERO and np.linalg.norm(self.translation) < TOL_MATRIX_ZERO:
            Ri = np.zeros((3, 1))
        else:
            Ri = self.matrix @ taui + self.translation 

        if np.linalg.norm(tauj) < TOL_MATRIX_ZERO and np.linalg.norm(self.translation) < TOL_MATRIX_ZERO:
            Rj = self.matrix @ Rvec
        else:
            Rj = self.matrix @ (tauj + Rvec) + self.translation

        Rnew = np.floor(Rj + TOL_MATRIX_ZERO) - np.floor(Ri + TOL_MATRIX_ZERO) 
        return Rnew

    def bra_cell_shift_cart(self, i, orbitals):
        """Cell shift of the transformed bra before it is reduced to the home cell."""
        from .map import formapsp

        _, _, taui, _ = formapsp(i, orbitals)
        cell_shift = np.floor(
            self.matrix @ taui + self.translation + TOL_MATRIX_ZERO
        )
        return self.lattice_cart @ cell_shift
    
@dataclass
class Mops(ops_actclass):
    time_reversal: bool 

    def __post_init__(self, permutation):
        self.rot_cart = permutation @ self.matrix @ np.linalg.inv(permutation)
        self.lattice_cart = permutation
        self.translation_cart = permutation @ self.translation
        
        self.U = rotget(self.rot_cart, self.spin_direction)
    @property
    def is_time_reversed(self) -> bool:

        return self.time_reversal

@dataclass
class Sops(ops_actclass):
    opSpin: np.ndarray
    
    def __post_init__(self, permutation):
        self.rev = True if np.linalg.det(self.opSpin) < 0 else False
        self.rot_cart = permutation @ self.matrix @ np.linalg.inv(permutation)
        self.lattice_cart = permutation
        self.translation_cart = permutation @ self.translation
       
        self.U = rotget(self.opSpin, self.spin_direction)
    @property
    def is_time_reversed(self) -> bool:
        return self.rev
        
    
