"""Some common utilities for creating the sky models"""

from pathlib import Path
import yaml
import numpy as np
from pyradiosky import SkyModel


# These paths and varaibles define some of the default directories and shared
# config files for H4C validation simulations
REPODIR = Path(__file__).parent.absolute()
SKYDIR = REPODIR / "sky_models"
CFGDIR = REPODIR / "config_files"
OUTDIR = REPODIR / "outputs"
HPCDIR = REPODIR / "hpc-configs"

LAYOUTDIR = CFGDIR / "array_layouts"

FULL_HERA_LAYOUT = LAYOUTDIR / "array_layout_hera_350.txt"

ANTS_DICT = {
    'H4C': np.genfromtxt(LAYOUTDIR / 'h4c_ants.txt')
    'HEX': np.arange(320),
    'FULL': np.arange(350),
    'HERA19': [0, 1, 2, 11, 12, 13, 14, 23, 24, 25, 26, 27, 37, 38, 39, 40, 52, 53, 54]
}

H4C_TELE_CONFIG = CFGDIR / "tele_config_hera_h4c.yaml"
H4C_ARRAY_LAYOUT = CFGDIR / "array_layout_hera_h4c.txt"

# H4C frequency information
with open(REPODIR / "h4c_freqs.yaml", "r") as fl:
    freq_info = yaml.load(fl, Loader=yaml.FullLoader)
H4C_FREQS = np.arange(
    freq_info["start"], freq_info["end"] + freq_info["delta"] / 2, freq_info["delta"]
)
H4C_CHANNEL_WIDTH = freq_info["delta"]

# These time parameters are default for all validation sims. It covers 24 LST hour.
VALIDATION_SIM_NTIMES = 17280
VALIDATION_SIM_INTEGRATION_TIME = 4.986347833333333
VALIDATION_SIM_START_TIME = 2458208.916228965

# HERA location - (latittude [deg], longitude [deg], height [m])
HERA_LOC = (-30.72152612068957, 21.428303826863015, 1051.6900000218302)

def make_hera_layout(name: str, ants: np.ndarray | None = None):
    
    if ants is None:
        ants = ANTS_DICT[name.upper()]

    direc = LAYOUTDIR / 'tmp'
    if not direc.exists():
        direc.mkdir()

    full_layout = np.sort(
        np.genfromtxt(FULL_HERA_LAYOUT, dtype=[int, float, float, float])
    )
    
    with open(direc / f"{name}.txt", 'w') as fl:
        fl.write("Name    Number  BeamID  E       N       U")
        for ant in ants:
            pos = full_layout[ant][1:]
            fl.write(f"HH{ant}\t{ant}\t0\t{pos[0]}\t{pos[1]}\t{pos[2]}")

    return fl

def make_tele_config(freq_interp_kind: str = 'cubic', spline_interp_order: int = 3):
    config = """
beam_paths:
  0: '{}'
telescope_location: (-30.72152612068957, 21.428303826863015, 1051.6900000218302)
telescope_name: HERA
freq_interp_kind: 'cubic'
spline_interp_opts:
        kx: 3
        ky: 3
"""
def write_sky(sky: SkyModel, model: str, channel: int):
    d = SKYDIR / model
    d.mkdir(parents=True, exist_ok=True)
    sky.write_skyh5(f"{d}/fch{channel:04d}.skyh5", clobber=True)
