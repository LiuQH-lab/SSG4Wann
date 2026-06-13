import os
import numpy as np 
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Any
from ..mpi.system import global_mpi_print
from ..exceptions import ConfigParseError

type RowVec = np.ndarray

@dataclass(slots=True)
class Config:
    seed: str = 'wannier90'
    soc: bool | None = None
    winpath: str = ''
    tb_mode: bool = False
    chnl: bool = True
    bands_trans: bool = False
    bands_num_points: int = 100
    kpath_segments: list[dict[str, Any]] = field(default_factory=list)
    hr4trans: str = ''
    NONCOLLINEAR_channel: bool | None = None
    each_symm: bool = False
    hard_ave: bool = False
    spin_direction: RowVec | None = None
    symm_output: bool = True
    extend_LatVec: bool = True
    forced_hermitianize: bool = False
    spinonly_speedup: bool = True

    def validate(self, mpi_print: Callable) -> None:
        if self.bands_trans and not self.kpath_segments:
            raise ConfigParseError("Error: 'bands_trans' is True, but no 'kpoint_path' block found in sg.in")
        if self.bands_trans and self.tb_mode:
            raise ConfigParseError(
                "Error: 'bands_trans' and 'tb_mode' cannot both be True."
            )

            
        if self.NONCOLLINEAR_channel is None:
            raise ConfigParseError("Error: NONCOLLINEAR_channel variable is not set.")

        if self.NONCOLLINEAR_channel == False and self.chnl == False:
            raise ConfigParseError("Error: Both NONCOLLINEAR_channel and chnl cannot be False. If you are doing collinear calculation, set chnl = True and NONCOLLINEAR_channel = False.")

        if not self.bands_trans:
            if self.spin_direction is None and self.NONCOLLINEAR_channel:
                mpi_print("Spin_direction variable is not set. Defaulting SAXIS to z-axis (0, 0, 1).")
                self.spin_direction = np.array([0.0, 0.0, 1.0])

            if self.each_symm and not self.hard_ave:
                raise ConfigParseError("Error: 'each_symm' is True but 'hard_ave' is False.")
            if self.hard_ave and not self.each_symm:
                mpi_print("Warning: 'hard_ave' is True. Large Errors possible!!!")
            if not self.NONCOLLINEAR_channel and self.spin_direction is None:
                raise ConfigParseError("Error: 'NONCOLLINEAR_channel' is False but 'spin_direction' is not set!!!") 
        else:
            if not self.hr4trans:
                raise ConfigParseError("Error: 'bands_trans' is True but 'use_hr_file' is not set.")

        
def _parse_bool(val: str) -> bool:
    return val.upper() in ('T', 'TRUE', '.TRUE.', 'true')


def infoload(config_path: str, rank: int) -> Config:
    mpi_print = partial(global_mpi_print, rank=rank)

    
    config = Config()
    reading_kpath = False 

    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', '!', '//')):
                    continue
                
                lower_line = line.lower()

                if 'begin kpoint_path' in lower_line:
                    reading_kpath = True
                    continue 
                if 'end kpoint_path' in lower_line:
                    reading_kpath = False
                    continue 

                if reading_kpath:
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            config.kpath_segments.append({
                                'label_start': parts[0],
                                'start': np.array(parts[1:4], dtype=float),
                                'label_end': parts[4],
                                'end': np.array(parts[5:8], dtype=float),
                            })
                        except ValueError:
                            mpi_print(f"Warning: Wrong K_path {line}")
                    continue 

                if '=' not in line:
                    continue
                

                key, val = (part.strip() for part in line.split('=', 1))
                val = val.strip("'").strip('"')

   
                match key.lower():
                    case 'seedname': config.seed = val
                    case 'use_win': config.winpath = val
                    case 'tb_mode': config.tb_mode = _parse_bool(val)
                    case 'use_hr_file': config.hr4trans = val
                    case 'bands_num_points': config.bands_num_points = int(val)
                    case 'soc': config.soc = _parse_bool(val)
                    case 'chnl': config.chnl = _parse_bool(val)
                    case 'bands_trans': config.bands_trans = _parse_bool(val)
                    case 'noncollinear_channel': config.NONCOLLINEAR_channel = _parse_bool(val)
                    case 'each_symm': config.each_symm = _parse_bool(val)
                    case 'hard_ave': config.hard_ave = _parse_bool(val)
                    case 'symm_output': config.symm_output = _parse_bool(val)
                    case 'spin_direction':
                        parts = val.split()
                        if len(parts) >= 3:
                            try:
                                config.spin_direction = np.array(parts[:3], dtype=float)
                            except ValueError:
                                raise ConfigParseError(f"Warning: Wrong spin_direction {line}")
                        else:
                            raise ConfigParseError(f"Warning: spin_direction needs 3 components in line: {line}")
                    case 'extend_latvec':
                        config.extend_LatVec = _parse_bool(val)
                    case 'forced_hermitianize':
                        config.forced_hermitianize = _parse_bool(val)
                    case 'spinonly_speedup':
                        config.spinonly_speedup = _parse_bool(val)
                        
    except FileNotFoundError:
        raise ConfigParseError(f"Error: Input file {config_path} not found.")


    
    config.validate(mpi_print)
    
    return config
