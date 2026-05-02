import argparse
import sys
import re
from pathlib import Path

from .parsergen.generate import get_sg_template
from .main import avg_kernel
from .version import __version__
from .mpi.system import mpi_init

def detect_system_settings(workdir: Path) -> dict:

    params = {
        "soc": "T",
        "seedname": "wannier90",
        "use_win": "wannier90.win",
        "noncollinear": "False"
    }
    
    incar_path = workdir / "INCAR"
    if incar_path.exists():
        content = incar_path.read_text(encoding="utf-8").upper()

        if re.search(r"LSORBIT\s*=\s*(?:\.TRUE\.|T)", content):
            params["soc"] = "T"

        elif re.search(r"LSORBIT\s*=\s*(?:\.FALSE\.|F)", content):
            params["soc"] = "F"
        else:
            raise ValueError("Failed to detect SOC setting from INCAR. Please ensure LSORBIT is set to .TRUE. or .FALSE.")
        if re.search(r"LNONCOLLINEAR\s*=\s*(?:\.TRUE\.|T)", content):
            params["noncollinear"] = "True"
        elif re.search(r"LNONCOLLINEAR\s*=\s*(?:\.FALSE\.|F)", content):
            params["noncollinear"] = "False"
        else:
            raise ValueError("Failed to detect noncollinear setting from INCAR. Please ensure LNONCOLLINEAR is set to .TRUE. or .FALSE.")
    else:
        raise FileNotFoundError(f"INCAR file not found in the working directory: {workdir}. Please make sure to place the INCAR file in the working directory or specify the correct path.")
    collinear_hr = list(workdir.glob("*.up_hr.dat"))
    if collinear_hr:

        seed = collinear_hr[0].name.replace(".up_hr.dat", "")
        if '_symmed' in seed.lower():
            seed = seed.replace('_symmed', '')
        params["seedname"] = seed
        params["use_win"] = f"{seed}.up.win"

    else:

        noncol_hr = list(workdir.glob("*_hr.dat"))
        if noncol_hr:
            seed = noncol_hr[0].name.replace("_hr.dat", "")
            if '_symmed' in seed.lower():
                seed = seed.replace('_symmed', '')
            params["seedname"] = seed
            params["use_win"] = f"{seed}.win"
    wann_path = workdir / params["use_win"]
    if not wann_path.exists():
        raise FileNotFoundError(f"Failed to detect the Wannier90 input file. Expected to find '{params['use_win']}' in the working directory: {workdir}. Please ensure the correct Wannier90 input file is present or specify the correct seedname in the config.")
    # else:

    return params




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssg4wann",
        description="SSG4Wann symmetrization tool",
    )
    parser.add_argument("-c", "--config", 
                        default="sg.in", 
                        help="Path to config file (default: sg.in)")
    parser.add_argument("-w", "--workdir", 
                        default=None, 
                        help="Working directory (default: directory of config)")
    parser.add_argument("--dry-run", 
                        action="store_true", 
                        help="Parse config and exit without running main workflow")
    parser.add_argument("--version", 
                        action="version", 
                        version=f"%(prog)s {__version__}")
    parser.add_argument("--init", 
                        action="store_true", 
                        help="Force initialize the working directory with a default sg.in file")
    return parser


def ssg4wann():
    
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
 
    if args.workdir is None:
        workdir = config_path.parent
    else:
        workdir = Path(args.workdir).expanduser().resolve()
    if args.init:
        if config_path.exists():
            print(f"[Warning] '{config_path.name}' already exists in {workdir}.")
            print("Initialization aborted to prevent overwriting.")
            sys.exit(1) 
        print(f"[Auto-Detect] Scanning {workdir} for system parameters... The ssg4wann will generate a default config /'sg.in' based on the detected parameters for you.")
        params = detect_system_settings(workdir)
        template_content = get_sg_template(params)

        try:
            config_path.write_text(template_content, encoding="utf-8")
            print(f"[Success] Generated configuration file at: {config_path}")
            print(f"          - SeedName: '{params['seedname']}'")
            print(f"          - SOC: {params['soc']}")
            print(f"          - Noncollinear: {params['noncollinear']}")
            if params['noncollinear'] == 'False':
                print(f"          Warning: Detected collinear system! Please specify the correct `spin_direction` in the generated config file to ensure correct symmetrization results.")
        except Exception as e:
            print(f"[Error] Failed to write config file: {e}")
            sys.exit(1)
        sys.exit(0)

    rank, comm, mpi_print, USE_MPI = mpi_init()
    if  not config_path.exists():
        mpi_print(f"[Auto-Detect] Scanning {workdir} for system parameters... The ssg4wann will generate a default config /'sg.in' based on the detected parameters for you.")
        params = detect_system_settings(workdir)
        template_content = get_sg_template(params)
        if rank == 0:
            try:
                config_path.write_text(template_content, encoding="utf-8")
                mpi_print(f"[Success] Generated configuration file at: {config_path}")
                mpi_print(f"          - SeedName: '{params['seedname']}'")
                mpi_print(f"          - SOC: {params['soc']}")
                mpi_print(f"          - Noncollinear: {params['noncollinear']}")
                if params['noncollinear'] == 'False':
                    mpi_print(f"          Warning: Detected collinear system! Please specify the correct `spin_direction` in the generated config file to ensure correct symmetrization results. The `ssg4wann` refuses to run with the default config for collinear systems to prevent incorrect symmetrization results!")
                    sys.exit(0)
            except Exception as e:
                mpi_print(f"[Error] Failed to write config file: {e}")
                sys.exit(1)
                
        if USE_MPI:
            comm.Barrier()
                


            

    if args.dry_run:
        mpi_print(f"[dry-run] config: {config_path}")
        mpi_print(f"[dry-run] workdir: {workdir}")
        sys.exit(0)


    mpi_print(f"Starting SSG4Wann with config: {config_path.name}")
    avg_kernel(rank,comm, mpi_print, USE_MPI, config_path=str(config_path))


if __name__ == "__main__":
    ssg4wann()