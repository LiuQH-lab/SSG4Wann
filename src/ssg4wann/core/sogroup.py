import numpy as np
from .constants import TOL_MATRIX_ZERO
from ..exceptions import GroupStructureError

def coset_decomposition(ops_list: list,
               TOL_MATRIX_ZERO: float = TOL_MATRIX_ZERO
               ) -> tuple[bool, list, list]:
    """
    input: the OSSG group operations list, not MSG
    output: bool: whether the order of the  spin-only group = 1
            list: the list of the nontrivial spin group operations
            list: the list of the spin-only group operations
    """
    G_SO = []
    G_NS = []

    seen_spatial_ops = set()
    for g in ops_list:
        R_spatial = np.array(g["real_rotation"])
        t_spatial = np.array(g["translation"])
        is_identity_rot = np.allclose(R_spatial, np.eye(3), atol=TOL_MATRIX_ZERO)
        is_zero_trans = np.allclose(t_spatial, np.zeros(3), atol=TOL_MATRIX_ZERO)

        if is_identity_rot and is_zero_trans:
            G_SO.append(g)

        r_flat = tuple(np.round(R_spatial.flatten(), decimals=4))
        t_flat = tuple(np.round(t_spatial.flatten(), decimals=4))
        spatial_key = r_flat + t_flat
        

        if spatial_key not in seen_spatial_ops:
            seen_spatial_ops.add(spatial_key)
            G_NS.append(g)
    is_real_matrix =  False if len(G_SO) == 1 else True
    order_SO = len(G_SO)
    order_NS = len(G_NS)
    if len(ops_list) != order_SO * order_NS:
        raise GroupStructureError(f"Error: The group structure is not as expected. Total ops: {len(ops_list)}, SO ops: {order_SO}, NS ops: {order_NS}. Expected total ops = SO ops * NS ops = {order_SO * order_NS}.")  
    return is_real_matrix, G_SO, G_NS