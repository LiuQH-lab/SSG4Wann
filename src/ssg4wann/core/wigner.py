import numpy as np
from itertools import permutations

def rotation_to_cubic_dmatrix(R_cart: np.ndarray, L: int) -> np.ndarray:
    """
    Transform rotation matrix to the expression in wannier orbital subspace.
    The sequence is:
    L=0: [s]
    L=1: [pz, px, py]
    L=2: [dz2, dxz, dyz, dx2-y2, dxy]
    """
    R = np.array(R_cart, dtype=float)
    
    if L == 0:
        D_cubic = np.array([[1.0]])
        
    elif L == 1:
        P = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)
        D_cubic = P @ R @ P.T

        
    elif L == 2:
        s3 = np.sqrt(3)
        M = [
            # 0: dz2 = (2z^2 - x^2 - y^2) / (2*sqrt(3))
            np.array([[-1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0,  2]]) / (2 * s3),
            # 1: dxz = x*z
            np.array([[ 0,  0,  1],
                      [ 0,  0,  0],
                      [ 1,  0,  0]]) / 2.0,
            # 2: dyz = y*z
            np.array([[ 0,  0,  0],
                      [ 0,  0,  1],
                      [ 0,  1,  0]]) / 2.0,
            # 3: dx2-y2 = (x^2 - y^2) / 2
            np.array([[ 1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0,  0]]) / 2.0,
            # 4: dxy = x*y
            np.array([[ 0,  1,  0],
                      [ 1,  0,  0],
                      [ 0,  0,  0]]) / 2.0
        ]
        
        D_cubic = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                D_cubic[i, j] = 2.0 * np.trace(M[i] @ R @ M[j] @ R.T)

    elif L == 3:

        f_polynomials = [
            {(0,0,3): 2, (2,0,1): -3, (0,2,1): -3}, 
            # 1: f_xz2 = x(5z^2 - r^2) = 4xz^2 - x^3 - xy^2
            {(1,0,2): 4, (3,0,0): -1, (1,2,0): -1}, 
            # 2: f_yz2 = y(5z^2 - r^2) = 4yz^2 - yx^2 - y^3
            {(0,1,2): 4, (2,1,0): -1, (0,3,0): -1}, 
            # 3: f_z(x2-y2) = z(x^2 - y^2) = zx^2 - zy^2
            {(2,0,1): 1, (0,2,1): -1},              
            # 4: f_xyz = xyz
            {(1,1,1): 1},                           
            # 5: f_x(x2-3y2) = x(x^2 - 3y^2) = x^3 - 3xy^2
            {(3,0,0): 1, (1,2,0): -3},              
            # 6: f_y(3x2-y2) = y(3x^2 - y^2) = 3x^2y - y^3
            {(2,1,0): 3, (0,3,0): -1}               
        ]


        T_f = []
        for poly in f_polynomials:
            T = np.zeros((3, 3, 3))
            for (px, py, pz), coeff in poly.items():
                indices = [0]*px + [1]*py + [2]*pz
                perms = list(set(permutations(indices)))
                val = coeff / len(perms)
                for p in perms:
                    T[p] = val
            T = T / np.linalg.norm(T)
            T_f.append(T)

        D_cubic = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                T_rot_j = np.einsum('au, bv, cw, uvw -> abc', R, R, R, T_f[j])
                D_cubic[i, j] = np.sum(T_f[i] * T_rot_j)
                

        
    else:
        raise ValueError(f"the function only supports L=0,1,2 but got L={L}")
    
    D_cubic[np.abs(D_cubic) < 1e-4] = 0.0
    det = np.linalg.det(D_cubic)
    if abs(det) - 1.0 > 1e-2:
        raise ValueError(f"the calculated D_cubic matrix of the rotation matrix {R_cart} has determinant {det}, which is not close to 1. d_cubic={D_cubic}. Please check the input rotation matrix.")
    return D_cubic



