from multiprocessing import Pool
import argparse
import os

import healpy as hp
import numpy as np
from pyradiosky import SkyModel
from astropy import units as u

import utils

# H4C frequencies
FREQ_ARRAY = utils.h4c_freqs


def make_confusion_map(f, nside=128):
    """Make a confusion map at a given frequency. Output NSIDE=128."""
    # Load source confusion data
    ref_freq = 154e6
    data_file = f"{utils.SKYDIR}/gleam_like/gleam_like_fainter.npz"

    with np.load(data_file) as data:
        flux_ref = data["flux_density"]
        colat = data["theta"]
        lon = data["phi"]
        spectral_index = data["spec_index"]

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
    confusion_map = (
        (confusion_map * u.Jy / u.sr)
        .to(u.K, equivalencies=u.brightness_temperature(f * u.Hz))
        .value
    )

    return confusion_map


def make_sky_model(fch):
    print(f"{os.getpid()}: computing channel ", fch)

    f = FREQ_ARRAY[fch]

    print(f"{os.getpid()}: getting confusion map ")
    confusion_map = make_confusion_map(f, nside=out_nside)

    # Build a SkyModel object
    # Output is per frequency, so the second dimension is 1.
    stokes = np.zeros((4, 1, out_npix)) * u.K
    stokes[0, :, :] = confusion_map * u.K
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
        "history": "Generate a confusion model with gleam-like sources.\n",
        "frame": "galactic",
    }
    confusion_model = SkyModel(**params)

    # Convert healpix to point
    confusion_model.healpix_to_point()

    # Transform coordinates of the GSM pixels from Galactic to ICRS.
    confusion_model.transform_to("icrs")

    utils.write_sky(confusion_model, f"confusion_nside{out_nside}", fch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Confusion SkyModel.")
    parser.add_argument(
        "--nside", type=int, default=128, help="output NSIDE of the healpix maps."
    )
    parser.add_argument("--ncores", type=int, default=None, help="number of processes.")
    args = parser.parse_args()

    out_nside = args.nside
    out_npix = hp.nside2npix(out_nside)

    with Pool(args.ncores) as p:
        p.map(make_sky_model, range(FREQ_ARRAY.size))
