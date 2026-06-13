from itertools import product
from ..core.ops_act import Mops, Sops
import numpy as np

from ..core.constants import TOL_MATRIX_ZERO
from ..exceptions import WannierMatchError, SpinRotationError
    
def calc_op(idx_op, soc, permutation, orbSpin, orbitals, hr_entry, spin_direction, config):
    """Calculate the explicit expression of each symmetry operation on the Wannier function subspaces and the generated Lattices."""
        
    idx, op = idx_op
    operator, conj_factor = markjudge(soc, op, permutation=permutation, spin_direction=spin_direction)
    repdict = operator.rep_find()

    local_op = [operator, conj_factor]
    local_actdict = {}
    local_LatSet = set()
    
    for wann in orbSpin:
        hopnew = operator.i_find(wann.global_index, repdict, orbSpin)
        local_actdict[(idx, wann.global_index)] = hopnew
    if config.extend_LatVec:
        for Rtu, block in hr_entry.items():
            for a, b in product(orbitals, orbitals):
                Lat = np.array(Rtu).reshape(3, 1)
                Latii = np.floor(operator.matrix @ (Lat + a.tau) + operator.translation) - np.floor(operator.matrix @ b.tau + operator.translation)
                LatNew = tuple(Latii.flat)
                local_LatSet.add(LatNew)
    else:
        for Rtu, block in hr_entry.items():
            local_LatSet.add(Rtu)
        
            
    
    return idx, local_op, local_actdict, local_LatSet

def calc_ent(R, num_wann, opset, actdict, hr_entry, orbSpin, nsymm, NONCOLLINEAR_channel):
    """Calculate the averaged symmetrized Hamiltonian matrix elements for each R and each pair of Wannier functions. """
    n_ops = len(opset)
    R_coords = tuple(int(x) for x in R)
    res = []

    if NONCOLLINEAR_channel:
        loop_ranges = [(range(1, num_wann + 1), range(1, num_wann + 1))]
    else:
        upid = range(1, num_wann + 1)
        dnid = range(num_wann + 1, 2 * num_wann + 1)
        loop_ranges = [(upid, upid), (dnid, dnid)]

    def get_entry_value(Rnew_key, i_raw, j_raw, i, j, operator):
        target_R = hr_entry.get(Rnew_key, {})
        if NONCOLLINEAR_channel:
            return target_R.get((i_raw, j_raw), 0)
        else:
            if i_raw > num_wann and j_raw > num_wann:
                return target_R.get((i_raw - num_wann, j_raw - num_wann), {}).get('dn', 0)
            elif i_raw <= num_wann and j_raw <= num_wann:
                return target_R.get((i_raw, j_raw), {}).get('up', 0)
            else:
                raise WannierMatchError(f"Error: spin index mismatch! the indices of the basis is i = {i}, j = {j}, after operated {operator} the new index is i = {i_raw}, j = {j_raw}. Check your spin_direction!!!")
    for i_range, j_range in loop_ranges:
        for i, j in product(i_range, j_range):
            entries_op = 0
            noe = 0
            for idx in range(n_ops):
                operator, conj_factor = opset[idx]
                Rnew = operator.R_find(i, j, R_coords, orbSpin)
                Rnew_key = tuple(Rnew.flat)

                if Rnew_key not in hr_entry:
                    noe += 1
                    
                    
                else:
                    proi, proj = actdict[(idx, i)], actdict[(idx, j)]
                    for (iNew, coei), (jNew, coej) in product(proi, proj):
                        entry = get_entry_value(Rnew_key, iNew, jNew, i, j, operator)
                        
                        if conj_factor:
                            entry = coei * entry.conjugate() * coej.conjugate()
                        else:
                            entry = coei.conjugate() * entry * coej
                        entries_op += entry.item() if hasattr(entry, 'item') else entry

            if entries_op != 0:
                entries_op /= (nsymm - noe)
            
            res.append([(*R_coords, int(i), int(j)), complex(entries_op)])

    return res



def calc_each(orig_idx, opset, LatSet, num_wann, actdict, hr_entry, orbSpin, NONCOLLINEAR_channel, nsymm):
    """
    Wrapper: Calculate the symmetrized Hamiltonian matrix elements for a single symmetry operation.
    """
    single_opset = [opset[orig_idx]]
    
    single_actdict = {}
    for (op_idx, orb_i), val in actdict.items():
        if op_idx == orig_idx:
            single_actdict[(0, orb_i)] = val
            
    op_results = []
    
    for R in LatSet:
        res_R = calc_ent(
            R=R, 
            num_wann=num_wann, 
            opset=single_opset,    
            actdict=single_actdict,
            hr_entry=hr_entry, 
            orbSpin=orbSpin, 
            nsymm=nsymm,       
            NONCOLLINEAR_channel=NONCOLLINEAR_channel
        )
        op_results.extend(res_R)
        
    return orig_idx, op_results


def calc_r_ent(R, num_wann, opset, actdict, r_entry, orbSpin, nsymm, NONCOLLINEAR_channel):
    """Group-average the Cartesian position-operator matrix elements for one R."""
    n_ops = len(opset)
    R_coords = tuple(int(x) for x in R)
    target_overlap = R_coords == (0, 0, 0)
    res = []

    if NONCOLLINEAR_channel:
        loop_ranges = [(range(1, num_wann + 1), range(1, num_wann + 1))]
    else:
        upid = range(1, num_wann + 1)
        dnid = range(num_wann + 1, 2 * num_wann + 1)
        loop_ranges = [(upid, upid), (dnid, dnid)]

    def get_entry_value(Rnew_key, i_raw, j_raw, i, j, operator):
        target_R = r_entry.get(Rnew_key, {})
        zero = np.zeros(3, dtype=complex)
        if NONCOLLINEAR_channel:
            return np.asarray(target_R.get((i_raw, j_raw), zero), dtype=complex)
        if i_raw > num_wann and j_raw > num_wann:
            return np.asarray(
                target_R.get((i_raw - num_wann, j_raw - num_wann), {}).get("dn", zero),
                dtype=complex,
            )
        if i_raw <= num_wann and j_raw <= num_wann:
            return np.asarray(
                target_R.get((i_raw, j_raw), {}).get("up", zero),
                dtype=complex,
            )
        raise WannierMatchError(
            f"Error: spin index mismatch! the indices of the basis is i = {i}, "
            f"j = {j}, after operated {operator} the new index is i = {i_raw}, "
            f"j = {j_raw}. Check your spin_direction!!!"
        )

    for i_range, j_range in loop_ranges:
        for i, j in product(i_range, j_range):
            entries_op = np.zeros(3, dtype=complex)
            noe = 0

            for idx in range(n_ops):
                operator, conj_factor = opset[idx]
                Rnew = operator.R_find(i, j, R_coords, orbSpin)
                Rnew_key = tuple(int(value) for value in Rnew.flat)
                if Rnew_key not in r_entry:
                    noe += 1
                    continue

                transformed_matrix_element = np.zeros(3, dtype=complex)
                bra_shift = operator.bra_cell_shift_cart(i, orbSpin).reshape(3)
                proi, proj = actdict[(idx, i)], actdict[(idx, j)]

                for (i_new, coei), (j_new, coej) in product(proi, proj):
                    entry = get_entry_value(
                        Rnew_key, i_new, j_new, i, j, operator
                    ).reshape(3)

                    # tb.dat stores the transformed states after reducing the bra
                    # to the home cell. Restore its absolute-cell position first.
                    if Rnew_key == (0, 0, 0) and i_new == j_new:
                        entry = entry + bra_shift

                    if conj_factor:
                        contribution = coei * entry.conjugate() * coej.conjugate()
                    else:
                        contribution = coei.conjugate() * entry * coej
                    transformed_matrix_element += contribution

                overlap = 1.0 if target_overlap and i == j else 0.0
                affine_part = (
                    transformed_matrix_element
                    - operator.translation_cart.reshape(3) * overlap
                )
                entries_op += np.linalg.solve(operator.rot_cart, affine_part)

            if n_effective := nsymm - noe:
                entries_op /= n_effective
            res.append([(*R_coords, int(i), int(j)), entries_op])

    return res


def calc_r_each(
    orig_idx,
    opset,
    LatSet,
    num_wann,
    actdict,
    r_entry,
    orbSpin,
    NONCOLLINEAR_channel,
    nsymm,
):
    """Calculate position matrix elements transformed by one symmetry operation."""
    single_opset = [opset[orig_idx]]
    single_actdict = {
        (0, orb_i): value
        for (op_idx, orb_i), value in actdict.items()
        if op_idx == orig_idx
    }
    op_results = []
    for R in LatSet:
        op_results.extend(
            calc_r_ent(
                R=R,
                num_wann=num_wann,
                opset=single_opset,
                actdict=single_actdict,
                r_entry=r_entry,
                orbSpin=orbSpin,
                nsymm=nsymm,
                NONCOLLINEAR_channel=NONCOLLINEAR_channel,
            )
        )
    return orig_idx, op_results

def markjudge(soc, op_data, permutation, spin_direction):
    """classify the symmetry operation and calculate the corresponding operator and the conjugation factor. The conjugation factor is used to determine whether the Hamiltonian matrix element should be conjugated when applying the symmetry operation. For M operations, the conjugation factor is determined by the time reversal property of the operation. For S operations, the conjugation factor is determined by the determinant of the spin rotation matrix. """
    
    if soc == True:
        if op_data['time_reversal'] == 1 or op_data['time_reversal'] == '1':
            time_reversal = False
        elif op_data['time_reversal'] == -1 or op_data['time_reversal'] == '-1':
            time_reversal = True
        else:
            raise SpinRotationError("Error time reversal value! The time reversal value must be either 1 (no time reversal) or -1 (with time reversal).")
        operator = Mops(
            matrix=np.array(op_data['real_rotation']),
            translation=np.array(op_data['translation']).reshape(3, 1),
            time_reversal=time_reversal,
            permutation=permutation,
            spin_direction=spin_direction
        )
        conj_factor = operator.time_reversal

        return operator, conj_factor
    
    elif soc == False:
        operator = Sops(
            matrix=np.array(op_data['real_rotation']),
            translation=np.array(op_data['translation']).reshape(3, 1),
            opSpin=np.array(op_data['spin_rotation']),
            permutation=permutation,
            spin_direction=spin_direction
        )
        det = np.linalg.det(operator.opSpin)
        if abs(det - 1) < TOL_MATRIX_ZERO:
            conj_factor = False
        elif abs(det + 1) < TOL_MATRIX_ZERO:
            conj_factor = True
        else:
            raise SpinRotationError("Error spin rotation!!! Check the determinant of the spin rotation matrix. Its determinant must be ±1. Check your input symmetry operation or structure file!!!")

        return operator, conj_factor
    





