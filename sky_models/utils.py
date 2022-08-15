"""Some common utilities for creating the sky models"""

from pathlib import Path
import yaml
import numpy as np

REPODIR = Path(__file__).parent.parent.absolute()
SKYDIR = Path(__file__).parent.absolute()

with open(REPODIR / 'freqs.yaml', 'r') as fl:
    freqinfo = yaml.load(fl, Loader=yaml.FullLoader)

freqs=np.arange(freqinfo['start'], freqinfo['end'], freqinfo['delta'])

def write(sky, model, channel: int):
    d = SKYDIR / model
    if not d.exists():
        d.mkdir(parents=True)

    sky.write_skyh5(f'{d}/fch{channel:04d}.skyh5', clobber=True)
