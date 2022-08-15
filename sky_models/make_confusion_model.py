"""Make source confusion SkyModel using healpy gridding."""
from multiprocessing import Pool, Array
import numpy as np
import healpy as hp
from pyradiosky import SkyModel
from astropy import units as u


# Load data
with np.load('./gleam_like/gleam_like_fainter.npz') as data:
    flux_ref = data['flux_density']
    colat = data['theta']
    lon = data['phi']
    spectral_index = data['spec_index']


# Some parameters
ref_freq = 154e6
freq_array = np.arange(1e8, 2e8, 97656.25)
nfreqs = freq_array.size
nside = 128
npix = hp.nside2npix(nside)

# Pixel indices of sources
pixel_inds = hp.ang2pix(nside, colat, lon, nest=False)

# Shared memory array for healpix fluxes
mp_arr = Array('d', nfreqs * npix)
# sum_flux = np.zeros((freq_array.size, npix)


# Multiprocessing over frequency, loop over pixel indices, 
# add source fluxes to healpix pixels
def assign_to_healpix(i):
    f = freq_array[i]
    flux_f = flux_ref * (f / ref_freq) ** spectral_index
    arr = np.frombuffer(mp_arr.get_obj())
    arr.shape = (nfreqs, npix)
    for j, p in enumerate(pixel_inds):
        arr[i, p] += flux_f[j]

        
with Pool() as p:
    p.map(assign_to_healpix, range(nfreqs))

    

# Load the sum_flux from shared memory
sum_flux = np.frombuffer(mp_arr.get_obj())
sum_flux.shape = (nfreqs, npix)

# Divde by pixel area to conver to Jy/sr
sum_flux = sum_flux / hp.nside2pixarea(nside)

# Save the array 
np.save('./gleam_like/gleam_like_fainter_healpix_sum_nside128.npy', sum_flux)

# This part kill the job maybe OOM
# # Build a SkyModel
# # Stokes must be in Jy/sr or K, so dividing by pixel area in sr
# stokes = np.zeros((4, nfreqs, npix)) * u.Jy / u.sr
# stokes[0, :, :] = sum_flux * u.Jy / u.sr
# params = {
#     'component_type': 'healpix',
#     'nside': nside,
#     'hpx_inds': np.arange(npix),
#     'hpx_order': 'ring',
#     'spectral_type': 'full',
#     'freq_array': freq_array * u.Hz,
#     'stokes': stokes,
#     'history': ' Generate a gleam-like confusion map.\n'
# }
# confusion_model = SkyModel(**params)

# # Convert to Kelvin
# confusion_model.jansky_to_kelvin()

# # Save
# confusion_model.write_skyh5('gleam_like_fainter_healpix_sum_nside128.sky5h',
#                             clobber=True)
