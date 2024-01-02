"""Some common utilities used throughout the core modules."""

from os import environ
from pathlib import Path

import numpy as np
import yaml

HPC = environ.get("VALIDATION_SYSTEM_NAME")

if HPC is None:
    raise OSError(
        "You must set the VALIDATION_SYSTEM_NAME environment variable to your system "
        "name, corresponding to a file in hpc-configs."
    )

# These paths and variables define some of the default directories and shared
# config files for H4C validation simulations
REPODIR = Path(__file__).parent.parent.absolute()
SKYDIR = REPODIR / "sky_models"
RAWSKYDIR = SKYDIR / "raw"
CFGDIR = REPODIR / "config_files"
OUTDIR = REPODIR / "outputs"
HPCDIR = REPODIR / "hpc-configs"
OBSPDIR = CFGDIR / "obsparams"
COMPRESSDIR = REPODIR / "compression-cache"
LOGDIR = REPODIR / "logs"


DIRFMT = "{sky_model}/nt17280-{chunks:03d}chunks-{layout}"
FLFMT = "{sky_model}_fch{fch:04d}_nt17280_chunk{ch:03d}_{layout}"

OBSPARAM_DIRFMT = DIRFMT
OBSPARAM_FLFMT = FLFMT

VIS_DIRFMT = DIRFMT
VIS_FLFMT = FLFMT

COMPRESS_FMT = "ch{chunks}_{layout_file}.npy"

with open(HPCDIR / f"{HPC}.yaml") as fl:
    HPC_CONFIG = yaml.load(fl, Loader=yaml.FullLoader)

LAYOUTDIR = CFGDIR / "array_layouts"

FULL_HERA_LAYOUT = LAYOUTDIR / "array_layout_hera_350.txt"

ANTS_DICT = {
    "H4C": np.genfromtxt(LAYOUTDIR / "h4c_ants.txt").astype(int),
    "HEX": np.arange(320),
    "FULL": np.arange(350),
    "HERA19": [0, 1, 2, 11, 12, 13, 14, 23, 24, 25, 26, 27, 37, 38, 39, 40, 52, 53, 54],
    "MINIMAL": [0, 1, 2, 4, 11, 23, 52],  # 3 N-S bls, 3 E-W bls (1,2,4 units)
}

# H4C frequency information
with open(REPODIR / "h4c_freqs.yaml") as fl:
    freq_info = yaml.load(fl, Loader=yaml.FullLoader)

_H4C_FREQS = np.arange(
    freq_info["start"], freq_info["end"] + freq_info["delta"] / 2, freq_info["delta"]
)

FREQS_DICT = {
    "H4C": _H4C_FREQS,
    "H6C": _H4C_FREQS,
}

# These time parameters are default for all validation sims. It covers 24 LST hour.
VALIDATION_SIM_NTIMES = 17280
VALIDATION_SIM_INTEGRATION_TIME = 4.986347833333333
VALIDATION_SIM_START_TIME = 2458208.916228965

# HERA location - (latittude [deg], longitude [deg], height [m])
HERA_LOC = (-30.72152612068957, 21.428303826863015, 1051.6900000218302)


def make_hera_layout(name: str, ants: np.ndarray | None = None) -> Path:
    """Create a HERA layout."""
    if ants is None:
        ants = ANTS_DICT[name.upper()]

    direc = LAYOUTDIR / "tmp"
    if not direc.exists():
        direc.mkdir()

    full_layout = np.genfromtxt(FULL_HERA_LAYOUT, skip_header=1)

    with open(direc / f"{name}.txt", "w") as fl:
        fl.write("Name    Number  BeamID  E       N       U\n")
        for ant in ants:
            pos = full_layout[ant][1:]
            fl.write(f"HH{ant}\t{ant}\t0\t{pos[0]}\t{pos[1]}\t{pos[2]}\n")

    return direc / f"{name}.txt"
