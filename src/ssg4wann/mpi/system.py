from __future__ import annotations  
import sys
import builtins
from functools import partial
import traceback
import time
from tqdm import tqdm
import os
from typing import Callable, Any

class RealTerminal:
    def __init__(self):
        self.file = None
        try:

            self.file = open('/dev/tty', 'w')
        except Exception:
            self.file = sys.stderr

    def write(self, text):
        self.file.write(text)

    def flush(self):
        self.file.flush()
        
    def close(self):
        if self.file is not sys.stderr:
            self.file.close()

def get_real_terminal():
    try:
        return open('/dev/tty', 'w')
    except Exception:
        return sys.stderr
    
def global_mpi_print(*args, rank: int, **kwargs):
    if rank == 0:
        kwargs.setdefault('flush', True)
        builtins.print(*args, **kwargs)

def mpi_init() -> tuple[int, Any, Callable, bool]:
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
            mpi_print(f"{total_processes - 1} worker processes (Rank 1 to {total_processes - 1}) are responsible for parallel computation.")
        else:
            mpi_print("You are running in serial mode without MPI parallelization. Consider using mpirun with multiple processes for faster symmetrization!")
        mpi_print("="*80)
        
    return rank, comm, mpi_print, USE_MPI


def mpi_excepthook(exc_type, exc_value, exc_traceback):
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        processor_name = MPI.Get_processor_name()
    except ImportError:
        comm = None
        rank = 0
        size = 1
        processor_name = "Localhost"

    error_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    sys.stderr.write("\n" + "!"*30 + " CRITICAL ERROR DETECTED " + "!"*30 + "\n")
    sys.stderr.write(f"Failing rank: {rank}/{size - 1} on host {processor_name}\n")
    sys.stderr.write(error_text)
    sys.stderr.write("!"*85 + "\n")
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(2) 

    if size > 1 and comm is not None:
        comm.Abort(1)
    else:
        sys.exit(1)


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
        
        results = [func(task) for task in pbar]
    
        if hasattr(out_terminal, 'close') and out_terminal is not sys.stderr:
            out_terminal.close()
            
        return results

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    chunk_size = len(iterable) // size
    remainder = len(iterable) % size
    
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)
    
    local_tasks = iterable[start:end]
    local_results = []
    
    # 进度条处理 (仅 Rank 0 显示)
    out_terminal = None
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
        
    # Rank 0 关闭终端句柄
    if rank == 0 and out_terminal is not None and hasattr(out_terminal, 'close') and out_terminal is not sys.stderr:
        out_terminal.close()
        
    gathered = comm.gather(local_results, root=0)

    if rank == 0:
        return [item for sublist in gathered for item in sublist]
    else:
        return None