"""
Create a specific realization of the EOR Gaussian Random Field.

This file was originally written by Zac Martinot, and called `make_nside256_realization.py`.
We have just updated it to take arguments and put files in the right paths.
https://github.com/zacharymartinot/redshifted_gaussian_fields
The version 
https://github.com/steven-murray/redshifted_gaussian_fields/tree/add-profiling
Is updated to use the cli.py commands including rgf. 
Make sure to install this repo. 
"""
from .slurm import slurmify
from . import utils

# this needs RM-512 for Nside >= 512
# @slurmify('grf-realization', time="1-12:00:00", partition='RM-512')
#@slurmify('grf-realization', time="0-04:00:00", partition='RM-shared', defaulttasks=128)  # for small
@slurmify('grf-realization',
          time="0-4:00:00", 
          defaultmem = "48GB",
          defaultnodes=1,
          defaulttasks=1*16,          # 48 was default 
          partition='hera',
          )
def run_compute_grf_realization(
    nside: int, seed: int = 2038, low_memory: bool = True,
):
    lmemstr = "--low-memory" if low_memory else "--no-low-memory"
    # Seed-tag the raw realization filename so different seeds coexist (covariance is
    # seed-independent and stays at raw/covariance.h5). Filename tag (not a subfolder)
    # because slurmify runs this locally via subprocess.call(cmd.split()), which can't
    # handle a `mkdir && ...` chain.
    return f"rgf realization --nside {nside} --seed {seed} {lmemstr} --covpath {utils.RAWSKYDIR / 'covariance.h5'} --outpath {utils.RAWSKYDIR / f'eor-grf-nside{nside}_seed{seed}.h5'} --overwrite"
    # return f"rgf realization --nside {nside} --seed {seed} {lmemstr} --covpath {utils.SKYDIR / 'raw' / 'covariance.h5'} --outpath {utils.SKYDIR / 'raw' / f'eor-grf-nside{nside}.h5'} --overwrite"
    
