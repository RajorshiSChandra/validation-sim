"""Some common utilities used throughout the core modules."""

import logging
from os import environ
from pathlib import Path
from parse import parse

import numpy as np
import yaml

logger = logging.getLogger(__name__)

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
BEAMDIR = REPODIR / "beams"

DIRFMT = "{sky_model}/{prefix}/nt17280-{chunks:05d}chunks-{layout}-{redundant}"
FLFMT = "fch{fch:04d}_chunk{ch:05d}"

def get_direc(sky_model:str, chunks: int, layout: str, redundant: bool, prefix: str = 'default'):
    return Path(
        DIRFMT.format(
        sky_model=sky_model, prefix=prefix, chunks=chunks, layout=layout,
        redundant="red" if redundant else 'nonred'
    ))
    
def get_file(chunk: int, channel: int, with_dir: bool = True, ext: str = None, **kw):
    stem = FLFMT.format(fch=channel, ch=chunk)

    fl = get_direc(**kw) / stem if with_dir else Path(stem)
    if ext:
        fl = fl.with_suffix(ext)
    return fl
    
def parse_fname(fname):
    return parse(FLFMT, fname).named

def parse_direc(direc: Path):
    parents = direc.parents
    if len(parents)>2:
        name = str(direc.relative_to(parents[2]))
    else:
        name = str(direc.relative_to(parents[1]))
    return parse(DIRFMT, name).named

def parse_path(path: Path, only_model: bool = False):
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")
    
    if path.is_file():
        if only_model:
            out = parse_direc(path.parent)
        else:
            out = parse_fname(path.stem) | parse_direc(path.parent)
    else:
        out = parse_direc(path)
    
    if not out:
        raise ValueError(f"path {path} did not adhere to any specifications")
    
COMPRESS_FMT = "ch{chunks}_{layout_file}.npy"

HPC = environ.get("VALIDATION_SYSTEM_NAME")

if HPC is None:
    logger.warning(
        "You must set the VALIDATION_SYSTEM_NAME environment variable to your system "
        "name, corresponding to a file in hpc-configs. Assuming local system.",
    )
    HPC_CONFIG = None
else:
    with open(HPCDIR / f"{HPC}.yaml") as fl:
        HPC_CONFIG = yaml.load(fl, Loader=yaml.FullLoader)

LAYOUTDIR = CFGDIR / "array_layouts"

FULL_HERA_LAYOUT = LAYOUTDIR / "array_layout_hera_350.txt"
IDEAL_HERA_LAYOUT = LAYOUTDIR / "array_layout_hera_350_ideal.txt"

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


def make_hera_layout(name: str, ants: np.ndarray | None = None, ideal: bool = True) -> Path:
    """Create a HERA layout."""
    if ants is None:
        ants = ANTS_DICT[name.upper()]

    direc = LAYOUTDIR / "tmp"
    if not direc.exists():
        direc.mkdir()

    full_layout = np.genfromtxt(IDEAL_HERA_LAYOUT if ideal else FULL_HERA_LAYOUT, skip_header=1)

    with open(direc / f"{name}.txt", "w") as fl:
        fl.write("Name    Number  BeamID  E       N       U\n")
        for ant in ants:
            pos = full_layout[ant][1:]
            fl.write(f"HH{ant}\t{ant}\t0\t{pos[0]}\t{pos[1]}\t{pos[2]}\n")

    return direc / f"{name}.txt"
