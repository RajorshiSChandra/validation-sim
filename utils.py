"""Some common utilities for creating the sky models"""

from pathlib import Path
import yaml
import numpy as np

REPODIR = Path(__file__).parent.parent.absolute()
SKYDIR = Path(__file__).parent.absolute()

with open(REPODIR / "h4c_freqs.yaml", "r") as fl:
    freq_info = yaml.load(fl, Loader=yaml.FullLoader)

h4c_freqs = np.arange(
    freq_info["start"], freq_info["end"] + freq_info["delta"] / 2, freq_info["delta"]
)


def write_sky(sky, model, channel: int):
    d = SKYDIR / model
    if not d.exists():
        d.mkdir(parents=True)

    sky.write_skyh5(f"{d}/fch{channel:04d}.skyh5", clobber=True)
