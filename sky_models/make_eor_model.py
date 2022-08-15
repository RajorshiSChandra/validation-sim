from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import h5py
import healpy as hp
from astropy import units as u
from pyradiosky import SkyModel


NFREQS = 1024
NSIDE = 128
NPIX = hp.nside2npix(NSIDE)

model_dir = '/ilifu/astro/projects/hera/Validation/h1c_idr3_sim/sky_models/eor1a'

f = h5py.File(
    f'{model_dir}/H1C_IDR3_validation_eor1a_seed_2983_nside_128.hdf5', 'r'
)

# Frequency -- unit Hz
freqs = f['nu'][:]

# HEALPix array -- dimension (nfreqs, npix) -- unit Jy/sr
hmaps = f['hmap'][:, :]

# Shift the maps values so there is no negative avalues
hmaps_min = hmaps.min(axis=1)[:, np.newaxis]
hmaps = hmaps - hmaps_min + 0.5 * np.abs(hmaps_min)


def make_eor_model(i):
    # Initialize Stoke array
    stokes = np.zeros((4, 1, NPIX)) * u.Jy / u.sr
    stokes[0, 0, :] = hmaps[i] * u.Jy / u.sr
    
    # Setup SkyModel params
    params = {
        'component_type': 'healpix',
        'nside': NSIDE,
        'hpx_order': 'ring',
        'hpx_inds': np.arange(NPIX),
        'spectral_type': 'full',
        'freq_array': [freqs[i]] * u.Hz,
        'stokes': stokes
    }
    
    eor_model = SkyModel(**params)
    eor_model.write_skyh5(
        f'{model_dir}/eor1a_seed2983_nside128_fch{i:04d}.skyh5', 
        clobber=True
    )
    

with Pool() as p:
    p.map(make_eor_model, range(NFREQS))
