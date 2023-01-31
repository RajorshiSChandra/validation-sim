from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import h5py
import healpy as hp
from astropy import units as u
from pyradiosky import SkyModel

here = Path(__file__).parent
model_dir = here / 'eor'
f = h5py.File(f'{model_dir}/healpix_maps.h5', 'r')

NSIDE = int(f['nside'][()])
NPIX = hp.nside2npix(NSIDE)

freqs = f['frequencies_mhz'][:] * 1e6  # units Hz
NFREQS = len(freqs)
seed = int(f['realization_random_seed'][()])

# HEALPix array -- dimension (nfreqs, npix) -- unit Jy/sr
hmaps = f['healpix_maps_Jy_per_sr'][:, :]

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
        f'{model_dir}/fch{i:04d}.skyh5', 
        clobber=True
    )
    

with Pool() as p:
    p.map(make_eor_model, range(NFREQS))
