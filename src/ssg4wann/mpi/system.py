import sys
import builtins
from functools import partial
import traceback
import time
from tqdm import tqdm
import os




def get_real_terminal():
    try:
        # Linux / MacOS
        return open('/dev/tty', 'w')
    except Exception:
        # Windows
        return sys.stderr
    
def global_mpi_print(*args, rank, **kwargs):
    if rank == 0:
        kwargs.setdefault('flush', True)
        builtins.print(*args, **kwargs)

def mpi_init() -> tuple[int, MPI.Comm, callable, bool]:
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        total_processes = comm.Get_size()
        USE_MPI = True if total_processes > 1 else False

    except ImportError:
        comm = None
        rank = 0
        total_processes = 1
        USE_MPI = False
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    if USE_MPI:
        sys.excepthook = mpi_excepthook
    
    mpi_print = partial(global_mpi_print, rank=rank)

    if rank == 0:
        welcome_art = r"""
================================================================================
__      __   _                    _       
\ \    / /__| |__ ___ _ __  ___  | |_ ___ 
 \ \/\/ / -_) / _/ _ \ '  \/ -_) |  _/ _ \
  \_/\_/\___|_\__\___/_|_|_\___|  \__\___/

 _____ _____ _____                          
/  ___/  ___|  __ \                         
\ `--.\ `--.| |  \/___ _   _ _ __ ___  _ __ ___  
 `--. \`--. \ | __/ __| | | | '_ ` _ \| '_ ` _ \ 
/\__/ /\__/ / |_\ \__ \ |_| | | | | | | | | | | |
\____/\____/ \____/___/\__, |_| |_| |_|_| |_| |_|
                        __/ |                    
                       |___/                     


            
              [ Developed by: NitreneG ]
================================================================================"""
        mpi_print(welcome_art)
        mpi_print("MPI successfully initialized!")
        mpi_print(f"MPI Parallel Symmetrization Starting with {total_processes} processes...")
        if total_processes > 1:
            mpi_print(f"1 master process (Rank 0) is responsible for coordination and I/O operations.")
            mpi_print(f"{total_processes} worker processes (Rank 0 to {total_processes - 1}) are responsible for parallel computation.")
        else:
            mpi_print(f"you are running in serial mode without MPI parallelization, consider using mpirun with multiple processes for faster symmetrization!")
        mpi_print("="*80)
    return rank, comm, mpi_print, USE_MPI

def mpi_excepthook(exc_type, exc_value, exc_traceback):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        error_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        sys.stderr.write("\n" + "!"*30 + " CRITICAL ERROR DETECTED " + "!"*30 + "\n")
        sys.stderr.write(f"Failing rank: {rank}/{size - 1} on host {MPI.Get_processor_name()}\n")
        sys.stderr.write(error_text)
        sys.stderr.write("!"*85 + "\n")
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(2) 

        if size > 1:
            comm.Abort(1)
        else:
            sys.exit(1)
        if should_emit:
            error_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            sys.stderr.write("\n" + "!"*30 + " CRITICAL ERROR DETECTED " + "!"*30 + "\n")
            sys.stderr.write(f"Failing rank: {rank}/{size - 1} on host {MPI.Get_processor_name()}\n")
            sys.stderr.write(error_text)
            sys.stderr.write("!"*85 + "\n")
            sys.stderr.flush()
def mpi_map(func, iterable, USE_MPI, comm=None, desc="Processing"):

    if not isinstance(iterable, list):
        try:
            iterable = sorted(list(iterable))
        except (TypeError, ValueError):
            iterable = list(iterable)

    is_cluster = any(env in os.environ for env in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID'])

    if not USE_MPI:
        if is_cluster:
            pbar = tqdm(
                iterable, 
                desc=f"[Serial] {desc}", 
                ascii=True, 
                total=len(iterable),
                file=sys.stdout,
                mininterval=10.0  
            )
        else:
            out_terminal = get_real_terminal()
            pbar = tqdm(
                iterable, 
                desc=f"[Serial] {desc}", 
                ascii=True, 
                total=len(iterable),
                file=out_terminal,   
                dynamic_ncols=True,
                mininterval=0.1      
            )
            

        return [func(task) for task in pbar]


    rank = comm.Get_rank()
    size = comm.Get_size()
    

    chunk_size = len(iterable) // size
    remainder = len(iterable) % size
    
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)
    
    local_tasks = iterable[start:end]
    local_results = []
    
    # 进度条处理 (仅 Rank 0 显示)
    if rank == 0:
        if is_cluster:
            pbar = tqdm(
                local_tasks, 
                desc=f"[Rank 0] {desc}", 
                ascii=True, 
                total=end-start,
                file=sys.stdout,
                mininterval=10.0  
            )
        else:
            out_terminal = get_real_terminal()
            pbar = tqdm(
                local_tasks, 
                desc=f"[Rank 0] {desc}", 
                ascii=True, 
                total=end-start,
                file=out_terminal,   
                dynamic_ncols=True,
                mininterval=0.1      
            )
    else:
        pbar = local_tasks
        
    for task in pbar:
        local_results.append(func(task))
        
    gathered = comm.gather(local_results, root=0)

    if rank == 0:
        return [item for sublist in gathered for item in sublist]
    else:
        return None