from multiprocessing import Pool
import argparse
import os

import healpy as hp
import numpy as np
from pyradiosky import SkyModel
from astropy import units as u
from pygdsm import GlobalSkyModel

import utils

# H4C frequencies
FREQ_ARRAY = utils.freqs


def make_gsm_map(f, smooth=True, nside=128):
    """Generate GSM map at the given frequency.
    Ouput is Galactic coordinates."""
    gsm = GlobalSkyModel(freq_unit="Hz")
    gsm_map = gsm.generate(f)

    m_npix = gsm_map.size
    m_nside = hp.npix2nside(m_npix)

    # Smooth with 1 deg Gaussian
    if smooth:
        gsm_map = hp.smoothing(gsm_map, fwhm=np.pi / 180, pol=False)

    # Downgrade from NSIDE=512. Default is 128, enough for HREA.
    # This preserve the flux integral which is what we need.
    # No further normalisation is required.
    if nside != m_nside:
        gsm_map = hp.ud_grade(gsm_map, nside)

    return gsm_map


def make_sky_model(fch):
    print(f"{os.getpid()}: computing channel ", fch)

    f = FREQ_ARRAY[fch]

    print(f"{os.getpid()}: getting gsm ")
    gsm_map = make_gsm_map(f, nside=out_nside)

    # Build a SkyModel object
    # The GSM map is Stoke I in Kelvin
    # Output is per frequency, so the second dimension is 1.
    stokes = np.zeros((4, 1, out_npix)) * u.K
    stokes[0, :, :] = gsm_map * u.K
    # SkyModel parameters.
    # The frame parameter defines the coordinate frame of the map.
    params = {
        "component_type": "healpix",
        "nside": out_nside,
        "hpx_inds": np.arange(out_npix),
        "hpx_order": "ring",
        "spectral_type": "full",
        "freq_array": f * u.Hz,
        "stokes": stokes,
        "history": "Generate a GSM2008 model.\n",
        "frame": "galactic",
    }
    gsm_model = SkyModel(**params)

    # Convert healpix to point
    gsm_model.healpix_to_point()

    # Transform coordinates of the GSM pixels from Galactic to ICRS.
    gsm_model.transform_to("icrs")

    utils.write_sky(gsm_model, f"gsm_nside{out_nside}", fch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GSM2008 SkyModel.")
    parser.add_argument(
        "--nside", type=int, default=128, help="output NSIDE of the healpix maps."
    )
    parser.add_argument("--ncores", type=int, default=None, help="number of processes.")
    args = parser.parse_args()

    out_nside = args.nside
    out_npix = hp.nside2npix(out_nside)

    with Pool(args.ncores) as p:
        p.map(make_sky_model, range(FREQ_ARRAY.size))
