"""
Create a specific realization of the EOR Gaussian Random Field.

This file was originally written by Zac Martinot, and called `make_nside256_realization.py`.
We have just updated it to take arguments and put files in the right paths.
"""
from redshifted_gaussian_fields import generator
import h5py
from . import utils
from .slurm import slurmify

def compute_grf_realization(nside: int, seed: int = 2038):
    covpath = utils.SKYDIR / 'raw' / 'covariance.h5'
    if not covpath.exists():
        raise RuntimeError('Covariance does not yet exist. Please run `vsim.py grf-covariance` to create it')

    gcfg = generator.restore_from_file(covpath)

    hmap = gcfg.generate_healpix_map_realization(seed, nside)

    map_save_file = utils.SKYDIR / 'raw' / f'eor-grf-nside{nside}.h5'

    with h5py.File(map_save_file, 'w') as h5f:
        h5f.create_dataset('frequencies_mhz', data=utils.FREQS_DICT['H6C']*1e-6)
        h5f.create_dataset('healpix_maps_Jy_per_sr', data=hmap)
        h5f.create_dataset('nside', data=nside)
        h5f.create_dataset('realization_random_seed', data=seed)

# this needs RM-512 for Nside >= 512
@slurmify('grf-realization', time="0-06:00:00", partition='RM-512')
def run_compute_grf_realization(
    nside: int, seed: int = 2038,
):
    return f"time python vsim.py grf-realization --local --nside {nside} --seed {seed}"
    