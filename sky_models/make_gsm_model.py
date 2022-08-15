import pathlib
from multiprocessing import Pool
import argparse
import healpy as hp
import numpy as np
import os

from pyradiosky import SkyModel
from astropy import units as u
from pygdsm import GlobalSkyModel


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
    # --------------------------------------------------

    # Smooth with 1 deg Gaussian
    if smooth:
        gsm_map = hp.smoothing(gsm_map, fwhm=np.pi/180, pol=False)

    # Downgrade from NSIDE=512. Default is 128, enough for HREA.
    # This preserve the flux integral which is what we need.
    # No further normalisation is required.
    if nside != m_nside:
        gsm_map = hp.ud_grade(gsm_map, nside)
    
    return gsm_map

    
def make_skymodel(fch):
    print(f"{os.getpid()}: computing channel ", fch)

    f = FREQ_ARRAY[fch]
    
    print(f"{os.getpid()}: getting gsm ")
    gsm_map = make_gsm_map(f, nside=OUT_NSIDE)
    
    # Build a SkyModel object
    # The GSM map is Stoke I in Kelvin
    # Output is per frequency, so the second dimension is 1.
    stokes = np.zeros((4, 1, OUT_NPIX)) * u.K
    stokes[0, :, :] = gsm_map * u.K
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
        'history': 'Generate a diffuse map with GSM2008.\n',
        'frame': 'galactic',
    }
    gsm_model = SkyModel(**params)
    
    # Convert healpix to point
    gsm_model.healpix_to_point()
    
    # Transform coordinates of the GSM pixels from Galactic to ICRS.
    gsm_model.transform_to('icrs')

    gsm_model.write_skyh5(f'{OUT_DIR}/gsm_nside{OUT_NSIDE}_fch{fch:04d}.skyh5',
                          clobber=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create GSM SkyModel')
    parser.add_argument('--out_nside', type=int, default=128,
                        help='output NSIDE of the healpix maps')
    parser.add_argument('--freq_chans', type=int, nargs='+', default=None,
                        help='Frequency channels to make maps. Default is None, which will generate maps for all 1024 frequency channels')    
    parser.add_argument('--out_dir', type=str, default='./',
                        help='Output directory. Output files will be "out_dir/gsm_nside<out_nside>_fch????.skyh5"')
    args = parser.parse_args()
    
    # Frequency array is fixed to HERA band
    FREQ_ARRAY = np.arange(1e8, 2e8, 97656.25)
    OUT_NSIDE = args.out_nside
    OUT_NPIX = hp.nside2npix(OUT_NSIDE)
    OUT_DIR = pathlib.Path(args.out_dir)
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = OUT_DIR.as_posix()
    
    if args.freq_chans is None:
        fchs = np.arange(FREQ_ARRAY.size)
    else:
        fchs = np.atleast_1d(np.array(args.freq_chans))

    with Pool() as p:
        p.map(make_skymodel, fchs)