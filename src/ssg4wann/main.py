import os
from functools import partial
from posixpath import abspath, dirname
import numpy as np

from findspingroup import find_spin_group_input_ssg
from findspingroup.find_spin_group import write_poscar_ssg_symmetry_dat
from .core.wannob import wannobs, WannOrb
from .parsergen import *
from .mpi import *


def avg_kernel(rank, comm, mpi_print, USE_MPI, config_path):

    workdir = dirname(abspath(config_path))


    
                
    #Parameter Initialization
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
        num_wann = None
        if rank == 0:
            hrob = hr(workdir, config.seed, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel)
            hr_entry, num_wann = hrob.hr_entry()
            POSCAR_gen(permutation, posi, os.path.join(workdir, 'INCAR'), config.spin_direction, config.NONCOLLINEAR_channel)
            ops_list = usegroup(config.soc, os.path.join(workdir, 'POSCAR'), config.symm_output)
            nsymm = len(ops_list)
        mpi_print(f"Finish loading the Group data: {nsymm} symmetry operations loaded")
        if USE_MPI:
            ops_list = comm.bcast(ops_list, root=0)
            nsymm = comm.bcast(nsymm, root=0)
            hr_entry, num_wann = comm.bcast((hr_entry, num_wann), root=0)
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
        spin_direction=config.spin_direction
        )
        mpi_print(f"Starting  operation expressions calculation...")

        resultsop = mpi_map(op_loop, enumerate(ops_list, start=0), USE_MPI, comm, desc="Parallel Operation")

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
            
            mpi_print(f"Starting parallel symmetrization ...")

            resultsent = mpi_map(ent_loop, LatSet, USE_MPI, comm, desc="Parallel Symmetrization")
            
            # write symmetrized entries
            mpi_print('finish analyzing, writing symmetrized hr file...')
            if rank == 0:
                Hsymm = [item for sublist in resultsent for item in sublist]
                outwrite(workdir, config.seed, reco = Hsymm, num_wann = num_wann, nrpts = nrptssymm, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel)
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

            resultshard = mpi_map(ent_loop, indices, USE_MPI, comm, desc="Parallel Symmetrization Each Symmetry")
    
            # write symmetrized entries
            mpi_print('finish analyzing, writing symmetrized hr operation...')
            if rank == 0:
                full_reco = []
                for item in resultshard:
                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
                        idx, raw_op_data = item
                    else:
                        idx = resultshard.index(item)
                        raw_op_data = item

                    flat_reco = []
                    
                    for element in raw_op_data:
                        if isinstance(element, list) and (len(element) != 2 or not isinstance(element[0], tuple)):
                            flat_reco.extend(element)
                        else:
                            flat_reco.append(element)

                    if config.each_symm == True:
                        outwrite(workdir, seed = f"op{idx+1}", reco = flat_reco, num_wann = num_wann, nrpts = nrptssymm, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel)

                    dict_reco = {}
                    for coords, H in flat_reco:
                        dict_reco[coords] = [H] 
                        full_reco.append(dict_reco)

                full_reco = aveterms(full_reco, nsymm)
                full_reco = [[coords, H] for coords, H in full_reco.items()]

                outwrite(workdir, seed = f"wannier90", reco = full_reco, num_wann = num_wann, nrpts = nrptssymm, NONCOLLINEAR_channel=config.NONCOLLINEAR_channel)
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
    mpi_print(f"Starting parallel band structure calculation...")

    resultseigen = mpi_map(eig_loop, k_points, USE_MPI, comm, desc="Parallel Band Structure Calculation")
    mpi_print(f"finished diagonalization, writing to {bandspath} ...")
    if rank == 0:
        eigenvalues = [item for sublist in resultseigen for item in sublist]
        eigenvalues = np.array(eigenvalues).reshape(len(k_points), num_wann)
        bandwrite(bandspath, x_axis, eigenvalues, hr4trans, labels)
        mpi_print("Band structure data generation finished!")

def usegroup(soc, POSCAR_path, symm_output):
    payload = find_spin_group_input_ssg(POSCAR_path)
    try:
        if soc == True:
            ops_list = payload["msg"]["ops"]

        elif soc == False:
            ops_list = payload["ssg"]["ops"]
        
    except Exception as e:
        raise ValueError(f"Failed to extract symmetry operations for soc '{soc}' from the group finding output. Please check the output POSCAR. Original error: {e}")

    if symm_output:
        write_poscar_ssg_symmetry_dat("ssg_symm.json", payload)
    return ops_list




   





