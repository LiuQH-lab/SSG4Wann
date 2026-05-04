import numpy as np

TOL_SPIN_DET = 1e-3
TOL_ROTATION_AXIS = 1e-2    
TOL_MATRIX_ZERO = 1e-4      
TOL_TENSOR_NORM = 1e-2      


TOL_WANNIER_MATCH = 1e-3    

sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0],
                    [0,-1]], dtype=complex)

DEFAULT_ORBITALS = {
    1: ['pz', 'px', 'py'],
    2: ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'],
    3: ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)']
}

PI = np.pi

