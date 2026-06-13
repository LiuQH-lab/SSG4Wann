import os
from functools import partial
from posixpath import abspath, dirname
import numpy as np
from findspingroup import find_spin_group_input_ssg
from findspingroup.find_spin_group import write_poscar_ssg_symmetry_dat
from .core.wannob import wannobs, WannOrb
from .parsergen import *
from .mpi import *
from .core.wannob import proj_seq
from .core.sogroup import coset_decomposition
from .exceptions import ConfigParseError


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


def avg_kernel(rank, comm, mpi_print, USE_MPI, config_path):

    workdir = dirname(abspath(config_path))
    config = infoload(config_path, rank)

    
    #Define Wannier orbitals 
    orbitals, permutation, permuK, posi = wannobs(os.path.join(workdir, config.winpath))
    orbSpin = []
    num_orbs = len(orbitals)
    for orb in orbitals:
        if config.chnl: 
            idx_up, idx_dn = orb.global_index, orb.global_index + num_orbs
        else:    
            idx_up, idx_dn = orb.global_index * 2 - 1, orb.global_index * 2
            
        orbSpin.append(WannOrb(orb.label, orb.L, orb.tau, idx_up, spin='up'))
        orbSpin.append(WannOrb(orb.label, orb.L, orb.tau, idx_dn, spin='dn'))
    orbSpin.sort(key=lambda x: x.global_index)
    
    #Hamiltonian symmetrization
    if config.bands_trans == False: 
        ops_list = None
        nsymm = None
        hr_entry = None
        r_entry = None
        tb_lattice = None
        num_wann = None
        obseq = {}
        
        if rank == 0:
            is_real_matrix = False
            if config.tb_mode:
                tbob = tb(
                    workdir,
                    config.seed,
                    NONCOLLINEAR_channel=config.NONCOLLINEAR_channel,
                )
                hr_entry, r_entry, num_wann = tbob.tb_entry()
                tb_lattice = tbob.lattice
                if not np.allclose(tb_lattice, permutation):
                    raise ConfigParseError(
                        "The lattice vectors in tb.dat and the selected .win file do not match."
                    )
            else:
                hrob = hr(workdir, config.seed, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel)
                hr_entry, num_wann = hrob.hr_entry()
            POSCAR_gen(permutation, posi, os.path.join(workdir, 'INCAR'), config.spin_direction, config.NONCOLLINEAR_channel, workdir)
            ops_list = usegroup(config.soc, os.path.join(workdir, 'POSCAR'), config.symm_output, workdir)

            if (
                config.spinonly_speedup
                and not config.tb_mode
                and not config.hard_ave
                and config.soc is False
            ):
                is_real_matrix, G_SO, G_NS = coset_decomposition(ops_list)
                ops_list = G_NS
            nsymm = len(ops_list)

            proj_seq(os.path.join(workdir, config.winpath))
        mpi_print(f"Finish loading the Group data: {nsymm} symmetry operations loaded")
        if USE_MPI:
            ops_list = comm.bcast(ops_list, root=0)
            nsymm = comm.bcast(nsymm, root=0)
            hr_entry, r_entry, tb_lattice, num_wann = comm.bcast(
                (hr_entry, r_entry, tb_lattice, num_wann), root=0
            )
            comm.barrier()


        #find entries
        LatSet = set()
        actdict = {}
        opset = {}
        
        op_loop = partial(
        calc_op, 
        soc=config.soc, 
        permutation=permutation, 
        orbSpin=orbSpin, 
        orbitals=orbitals, 
        hr_entry=hr_entry,
        spin_direction=config.spin_direction,
        config=config
        )
        mpi_print(f"Starting  operation expressions calculation...")

        resultsop = mpi_map(op_loop, enumerate(ops_list, start=0), USE_MPI, comm, desc="Operation Processing")

        actdict = {}
        LatSet = set()
        if rank == 0:
            for idx, local_op_data, local_actdict, local_LatSet in resultsop:
                opset[idx] = local_op_data
                actdict.update(local_actdict)
                LatSet.update(local_LatSet)

        if USE_MPI:
            actdict = comm.bcast(actdict, root=0)
            LatSet = comm.bcast(LatSet, root=0)
            opset = comm.bcast(opset, root=0)

        nrptssymm = len(LatSet)

        if config.hard_ave == False:
            ent_loop = partial(
            calc_ent, 
            num_wann=num_wann, 
            opset=opset, 
            actdict=actdict, 
            hr_entry=hr_entry, 
            orbSpin=orbSpin, 
            nsymm=nsymm, 
            NONCOLLINEAR_channel=config.NONCOLLINEAR_channel
            )
            
            mpi_print(f"Starting symmetrization ...")

            resultsent = mpi_map(ent_loop, LatSet, USE_MPI, comm, desc="Symmetrization processing")
            
            # write symmetrized entries
            if rank == 0:
                Hsymm = [item for sublist in resultsent for item in sublist]

                if is_real_matrix:
                    for ent in Hsymm:
                        ent[1] = ent[1].real

                if config.forced_hermitianize:
                    print("analyzing Hermitian symmetrization results...")
                    tot_num_wann = num_wann if config.NONCOLLINEAR_channel else num_wann * 2
                    
                    Hsymm = hr.hermitize_hr(Hsymm, tot_num_wann)
                if not config.tb_mode:
                    print('finish analyzing, writing symmetrized hr file...')
                    outwrite(workdir, config.seed, reco = Hsymm, num_wann = num_wann, nrpts = nrptssymm, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel, chnl = config.chnl)

            if config.tb_mode:
                r_loop = partial(
                    calc_r_ent,
                    num_wann=num_wann,
                    opset=opset,
                    actdict=actdict,
                    r_entry=r_entry,
                    orbSpin=orbSpin,
                    nsymm=nsymm,
                    NONCOLLINEAR_channel=config.NONCOLLINEAR_channel,
                )
                mpi_print("Starting position-matrix symmetrization ...")
                results_r = mpi_map(
                    r_loop,
                    LatSet,
                    USE_MPI,
                    comm,
                    desc="Position Matrix Symmetrization",
                )
                if rank == 0:
                    rsymm = [item for sublist in results_r for item in sublist]
                    if config.forced_hermitianize:
                        total_wann = (
                            num_wann
                            if config.NONCOLLINEAR_channel
                            else num_wann * 2
                        )
                        rsymm = tb.hermitize_r(rsymm, total_wann)
                    tb.outwrite(
                        workdir,
                        config.seed,
                        tb_lattice,
                        Hsymm,
                        rsymm,
                        num_wann,
                        config.NONCOLLINEAR_channel,

                    )
            mpi_print('Symmetrization finished!')


        elif config.hard_ave == True:
            indices = list(range(len(opset)))

            ent_loop = partial(
            calc_each, 
            opset=opset,
            LatSet=LatSet,
            num_wann=num_wann, 
            actdict=actdict, 
            hr_entry=hr_entry, 
            orbSpin=orbSpin, 
            nsymm=1, 
            NONCOLLINEAR_channel=config.NONCOLLINEAR_channel
            )
            mpi_print(f"Starting parallel symmetrization ...")

            resultshard = mpi_map(ent_loop, indices, USE_MPI, comm, desc="Symmetrization Each Symmetry Processing")
    
            # write symmetrized entries
            mpi_print('finish analyzing symmetrized Hamiltonian operations...')
            if rank == 0:
                Hsymm, H_per_operation = _average_operation_results(
                    resultshard, nsymm
                )
                if not config.tb_mode:
                    if config.each_symm:
                        for idx, records in H_per_operation.items():
                            outwrite(workdir, seed = f"op{idx+1}", reco = records, num_wann = num_wann, nrpts = nrptssymm, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel, chnl = config.chnl)
                    outwrite(workdir, seed = f"wannier90", reco = Hsymm, num_wann = num_wann, nrpts = nrptssymm, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel, chnl = config.chnl)

            if config.tb_mode:
                r_loop = partial(
                    calc_r_each,
                    opset=opset,
                    LatSet=LatSet,
                    num_wann=num_wann,
                    actdict=actdict,
                    r_entry=r_entry,
                    orbSpin=orbSpin,
                    nsymm=1,
                    NONCOLLINEAR_channel=config.NONCOLLINEAR_channel,
                )
                results_r_hard = mpi_map(
                    r_loop,
                    indices,
                    USE_MPI,
                    comm,
                    desc="Position Matrix Each Symmetry Processing",
                )
                if rank == 0:
                    rsymm, r_per_operation = _average_operation_results(
                        results_r_hard, nsymm
                    )
                    if config.forced_hermitianize:
                        total_wann = (
                            num_wann
                            if config.NONCOLLINEAR_channel
                            else num_wann * 2
                        )
                        Hsymm = hr.hermitize_hr(Hsymm, total_wann)
                        rsymm = tb.hermitize_r(rsymm, total_wann)
                    if config.each_symm:
                        for idx in sorted(H_per_operation):
                            tb.outwrite(
                                workdir,
                                f"op{idx+1}",
                                tb_lattice,
                                H_per_operation[idx],
                                r_per_operation[idx],
                                num_wann,
                                config.NONCOLLINEAR_channel,
                                config.tb_precision,
                            )
                    tb.outwrite(
                        workdir,
                        config.seed,
                        tb_lattice,
                        Hsymm,
                        rsymm,
                        num_wann,
                        config.NONCOLLINEAR_channel,
                        config.tb_precision,
                    )
            mpi_print('Symmetrization finished!')

    elif config.bands_trans == True:  #bands transformation
        hrob = None
        if rank == 0:
            hrob = hr(workdir, config.seed, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel, hr4trans=config.hr4trans)
        if USE_MPI:
            hrob = comm.bcast(hrob, root = 0)
        bds_trans(hrob, workdir, config.seed, config.hr4trans, config.bands_num_points, config.kpath_segments, permuK, permutation, comm, USE_MPI)

def bds_trans(hrob, workdir, seed, hr4trans, bands_num_points, kpath, permuK, permutation, comm, USE_MPI):
    if USE_MPI:
        rank = comm.Get_rank()
    else:
        rank = 0
    mpi_print = partial(global_mpi_print, rank=rank)
    mpi_print(f'transforing {hr4trans} to band structure data...')
    key = hr4trans.split('.')[0]
    bandspath = os.path.join(workdir, key + '_bands.dat')
    k_points, x_axis, labels = hrob.Kpoints_gen(bands_num_points, kpath, permuK)
    hr_entry, num_wann = hrob.hr_entry()

    eig_loop = partial(
        hrob.hr2bds, 
        num_wann=num_wann,
        hr_entry=hr_entry,
        permuK=permuK, 
        permutation=permutation
    )
    mpi_print(f"Starting band structure calculation...")

    resultseigen = mpi_map(eig_loop, k_points, USE_MPI, comm, desc="Band Structure Calculation")
    mpi_print(f"finished diagonalization, writing to {bandspath} ...")
    if rank == 0:
        eigenvalues = [item for sublist in resultseigen for item in sublist]
        eigenvalues = np.array(eigenvalues).reshape(len(k_points), num_wann)
        bandwrite(bandspath, x_axis, eigenvalues, hr4trans, labels)
        mpi_print("Band structure data generation finished!")

def usegroup(soc, POSCAR_path, symm_output, workdir):
    payload = find_spin_group_input_ssg(POSCAR_path)
    try:
        if soc == True:
            ops_list = payload["msg"]["ops"]

        elif soc == False:
            ops_list = payload["ssg"]["ops"]
        
    except Exception as e:
        raise ConfigParseError(f"Failed to extract symmetry operations for soc '{soc}' from the group finding output. Please check the output POSCAR. Original error: {e}")

    if symm_output:
        write_poscar_ssg_symmetry_dat(os.path.join(workdir, "ssg_symm.json"), payload)
    return ops_list




   


