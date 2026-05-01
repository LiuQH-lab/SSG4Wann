import numpy as np

def revmapsp(labelNew, L, tau, spin, orbitals, atol=0.01):
    """information to index"""
    target_tau = np.array(tau, dtype=float).reshape(3)
    target_tau = target_tau % 1.0 

    for orb in orbitals:
        if orb.label != labelNew:
            continue
        if orb.spin != spin:
            continue
        current_tau = np.array(orb.tau, dtype=float).reshape(3) % 1.0
        diff = np.abs(current_tau - target_tau)
        is_match = True
        for i in range(3):
            d = diff[i]
            if not (d < atol or np.abs(d - 1.0) < atol):
                is_match = False
                break 
        if is_match:
            return orb.global_index

    return None
        
def formapsp(i, orbitals):
    """index to information"""
    orb = orbitals[i - 1]
    return orb.label, orb.L, orb.tau, orb.spin
    