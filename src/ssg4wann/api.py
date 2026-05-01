from pathlib import Path

from .mpi.system import mpi_init
from .cli import detect_system_settings, get_sg_template
from .main import avg_kernel

def quick_run(workdir: str = ".", config_name: str = "sg.in") -> None:

    workdir_path = Path(workdir).expanduser().resolve()
    config_path = workdir_path / config_name
    rank, comm, mpi_print, USE_MPI = mpi_init()
    if not config_path.exists():
        mpi_print(f"[API] cannot find '{config_name}' in {workdir_path}. Attempting to auto-generate /'sg.in' based on detected system parameters...")
        try:
            params = detect_system_settings(workdir_path)
            template_content = get_sg_template(params)
            config_path.write_text(template_content, encoding="utf-8")
            mpi_print(f"[API] Configuration file generated successfully: {config_path.name}")
        except Exception as e:

            raise RuntimeError(f"failed to generate configuration file: {e}")


    mpi_print(f"[API] Starting SSG4Wann with config: {config_path.name} in {workdir_path}...")

    avg_kernel(rank, comm, mpi_print, USE_MPI, config_path=str(config_path))