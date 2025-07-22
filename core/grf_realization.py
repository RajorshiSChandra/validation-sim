"""
Create a specific realization of the EOR Gaussian Random Field.

This file was originally written by Zac Martinot, and called `make_nside256_realization.py`.
We have just updated it to take arguments and put files in the right paths.
"""
from .slurm import slurmify
from . import utils

# this needs RM-512 for Nside >= 512
@slurmify('grf-realization', time="1-12:00:00", partition='RM-512')
#@slurmify('grf-realization', time="0-04:00:00", partition='RM-shared', defaulttasks=128)  # for small
def run_compute_grf_realization(
    nside: int, seed: int = 2038, low_memory: bool = True,
):
    lmemstr = "--low-memory" if low_memory else "--no-low-memory"
    return f"rgf realization --nside {nside} --seed {seed} {lmemstr} --covpath {utils.SKYDIR / 'raw' / 'covariance.h5'} --outpath {utils.SKYDIR / 'raw' / f'eor-grf-nside{nside}.h5'} --overwrite"
    
