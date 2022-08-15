from multiprocessing import Pool
import argparse
import healpy as hp
import numpy as np
import os

from pyradiosky import SkyModel
from astropy import units as u
from pygdsm import GlobalSkyModel

    
def make_confusion_map(f, nside=128):
    """Make a confusion map at a given frequency. Output NSIDE=128."""
    # Load source confusion data
    ref_freq = 154e6
    data_file = './gleam_like/gleam_like_fainter.npz'
    
    with np.load(data_file) as data:
        flux_ref = data['flux_density']
        colat = data['theta']
        lon = data['phi']
        spectral_index = data['spec_index']

    # Pixel indices of confusion sources
    pixel_inds = hp.ang2pix(nside, colat, lon, nest=False)
    
    # Power law interpolation to get source fluxes at the given frequency.
    flux_f = flux_ref * (f / ref_freq) ** spectral_index

    # Initialize confusion map
    confusion_map = np.zeros(hp.nside2npix(nside))
    
    # Loop over pixel indices of the sources and add fluxes to the right pixels.
    for i, p in enumerate(pixel_inds):
        confusion_map[p] += flux_f[i]
    
    # Divde by pixel area to conver to Jy/sr
    confusion_map = confusion_map / hp.nside2pixarea(nside)
    
    # Now convert from Jy/sr to kelvin
    confusion_map = (confusion_map * u.Jy / u.sr).to(
        u.K, equivalencies=u.brightness_temperature(f*u.Hz)
    ).value
    
    return confusion_map

        
def make_gsm_map(f, smooth=True, nside=128):
    """Generate GSM map at the given frequency. 
    Ouput is Galactic coordinates."""
    gsm = GlobalSkyModel(freq_unit='Hz')
    gsm_map = gsm.generate(f)

    m_npix = gsm_map.size
    m_nside = hp.npix2nside(m_npix)
    
    # Commented out as this does not work properly and 
    # pyradiosky can now handle coordinate frame
    # ------------------------------------------------
    # Rotate to Celestial
    # Simplistic rotation copied from pygdsm
    # Improvement can be made here if needed
    # ipix_g = np.arange(m_npix)
    # coordrotate = hp.Rotator(coord=['C', 'G'], inv=True)
    # theta_g, phi_g = hp.pix2ang(m_nside, ipix_g)
    # theta_c, phi_c = coordrotate(theta_g, phi_g)
    # ipix_c = hp.ang2pix(m_nside, theta_c, phi_c)
    # gsm_map = gsm_map[ipix_c]
    --------------------------------------------------

    # Smooth with 1 deg Gaussian
    if smooth:
        gsm_map = hp.smoothing(gsm_map, fwhm=np.pi/180, pol=False)

    # Downgrade from NSIDE=512. Default is 128, enough for HREA.
    # This preserve the flux integral which is what we need.
    # No further normalisation is required.
    if nside != m_nside:
        gsm_map = hp.ud_grade(gsm_map, nside)
    
    return gsm_map
        
    
def make_map(fch):
    print(f"{os.getpid()}: computing channel ", fch)

    f = FREQ_ARRAY[fch]
    
    print(f"{os.getpid()}: getting gsm ")
    gsm_map = make_gsm_map(f, nside=OUT_NSIDE)

    print(f"{os.getpid()}: getting confusion map ")
    confusion_map = make_confusion_map(f, nside=OUT_NSIDE)
    # Save maps for debugging
    np.save(f'gsm2008_smooth1deg_nside128_fch{fch:04d}.npy', gsm_map)
    np.save(f'gleam_like_confusion_nside128_fch{fch:04d}..npy', confusion_map)
    diffuse_map = gsm_map + confusion_map
    
    # Build a SkyModel object
    # The GSM map is Stoke I in Kelvin
    # Output is per frequency, so the second dimension is 1.
    stokes = np.zeros((4, 1, OUT_NPIX)) * u.K
    stokes[0, :, :] = diffuse_map * u.K
    # SkyModel parameters.
    # The frame parameter defines the coordinate frame of the map.
    params = {
        'component_type': 'healpix',
        'nside': OUT_NSIDE,
        'hpx_inds': np.arange(OUT_NPIX),
        'hpx_order': 'ring',
        'spectral_type': 'full',
        'freq_array': f * u.Hz,
        'stokes': stokes,
        'history': 'Generate a diffuse map with GSM2008 and gleam-like confusion sources.\n',
        'frame': 'galactic',
    }
    diffuse_model = SkyModel(**params)
    
    # Convert healpix to point
    diffuse_model.healpix_to_point()
    
    # Transform coordinates of the GSM pixels from Galactic to ICRS.
    diffuse_model.transform_to('icrs')

    diffuse_model.write_skyh5(f'{OUT_DIR}/diffuse_nside{OUT_NSIDE}_fch{fch:04d}.skyh5',
                               clobber=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create GSM + Diffuse maps.')
    parser.add_argument('--out_nside', type=int, default=128,
                        help='output NSIDE of the healpix maps.')
    parser.add_argument('--freq_chans', type=int, nargs='+', default=None,
                        help='Frequency channels to make maps. Default is None, which will generate maps for all 1024 frequency channels.')    
    parser.add_argument('--out_dir', type=str, default='./',
                        help='Output directory. Output files will be "out_dir/diffuse_nside<out_nside>_fch????.skyh5".')
    args = parser.parse_args()
    
    # Frequency array is fixed to HERA band
    FREQ_ARRAY = np.arange(1e8, 2e8, 97656.25)
    OUT_NSIDE = args.out_nside
    OUT_NPIX = hp.nside2npix(OUT_NSIDE)
    OUT_DIR = args.out_dir
    
    if args.freq_chans is None:
        fchs = np.arange(FREQ_ARRAY.size)
    else:
        fchs = np.atleast_1d(np.array(args.freq_chans))

    with Pool() as p:
        p.map(make_map, fchs)
