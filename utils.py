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


def write_sky(sky: SkyModel, model: str, channel: int):
    d = SKYDIR / model
    d.mkdir(parents=True, exist_ok=True)
    sky.write_skyh5(f"{d}/fch{channel:04d}.skyh5", clobber=True)
