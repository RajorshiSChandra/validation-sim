import os
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import healpy as hp
import click
import h5py
from astropy.table import Table
from astropy.coordinates import Longitude, Latitude
from astropy import units
from astropy.units import Quantity
from pyradiosky import SkyModel
from pygdsm import GlobalSkyModel

import utils

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=100)

H4C_FREQS = utils.H4C_FREQS * units.Hz
GL_MODEL_FILE = utils.SKYDIR / "gleam_like.skyh5"
ATEAM_MODEL_FILE = utils.SKYDIR / "ateam.skyh5"


"""===LOGIC CODE==="""

# Commented out due to missing source file
# def make_eor_model(fch):
#     """Make EoR SkyModel."""
#     model_dir = utils.SKYDIR / 'eor'
#     f = h5py.File(f'{model_dir}/healpix_maps.h5', 'r')

#     nside = int(f['nside'][()])
#     npix = hp.nside2npix(nside)

#     freqs = f['frequencies_mhz'][:] * 1e6  # units Hz
#     # nfreqs = len(freqs)
#     # seed = int(f['realization_random_seed'][()])

#     # HEALPix array -- dimension (nfreqs, npix) -- unit Jy/sr
#     hmaps = f['healpix_maps_Jy_per_sr'][:, :]

#     # Shift the maps values so there is no negative avalues
#     hmaps_min = hmaps.min(axis=1)[:, np.newaxis]
#     hmaps = hmaps - hmaps_min + 0.5 * np.abs(hmaps_min)

#     # Initialize Stoke array
#     stokes = np.zeros((4, 1, npix)) * units.Jy / units.sr
#     stokes[0, 0, :] = hmaps[fch] * units.Jy / units.sr

#     # Setup SkyModel params
#     params = {
#         'component_type': 'healpix',
#         'nside': nside,
#         'hpx_order': 'ring',
#         'hpx_inds': np.arange(npix),
#         'spectral_type': 'full',
#         'freq_array': [freqs[fch]] * units.Hz,
#         'stokes': stokes
#     }

#     eor_model = SkyModel(**params)
#     eor_model.write_skyh5(
#         f'{model_dir}/fch{fch:04d}.skyh5',
#         clobber=True
#     )


def make_gleam_like_model():
    data_file = f"{utils.SKYDIR}/gleam_like/gleam_like_brighter.npz"

    with np.load(data_file) as data:
        flux_ref = data["flux_density"]
        colat = data["theta"]
        lon = data["phi"]
        spectral_index = data["spec_index"]

    # Put synthetic sources into a SkyModel
    nsources = len(flux_ref)

    # Names are required for point sources
    names = [f"gl{i:06d}" for i in range(nsources)]

    # TODO: Change to SkyCoord with frame keyword required by pyradiosky=>0.3.0
    ra = Longitude(lon * units.rad)
    dec = Latitude(((np.pi / 2) - colat) * units.rad)

    stokes = np.zeros((4, 1, nsources)) * units.Jy
    stokes[0, :, :] = flux_ref * units.Jy

    # Reference frequency is 154 MHz
    ref_freq = 154e6 * np.ones(nsources) * units.Hz

    gl_model_params = {
        "name": names,
        "ra": ra,
        "dec": dec,
        "stokes": stokes,
        "spectral_type": "spectral_index",
        "reference_frequency": ref_freq,
        "spectral_index": spectral_index,
        "history": "Load gleam-like sources\n",
    }

    gl_model = SkyModel(**gl_model_params)
    gl_model.write_skyh5(GL_MODEL_FILE, clobber=False)


def make_ateam_model():
    """Make a SkyModel for sources peeled from GLEAM."""
    # The first 9 sources are peeled from GLEAM (Hurley-Walker+2017)
    names = [
        "3C444",
        "CentaurusA",
        "HydraA",
        "PictorA",
        "HerculesA",
        "VirgoA",
        "Crab",
        "CygnusA",
        "CassiopeiaA",
        "FornaxA",
    ]

    # RA/Dec.
    # For Fornax A, represent it as a single source using its host galaxy position
    # TODO: Change to SkyCoord with frame keyword required by pyradiosky=>0.3.0
    ra = Longitude(
        [
            "22h14m16s",
            "13h25m28s",
            "09h18m06s",
            "05h19m50s",
            "16h51m08s",
            "12h30m49s",
            "05h34m32s",
            "19h59m28s",
            "23h23m28s",
            "3h22m42s",
        ]
    )
    dec = Latitude(
        [
            "-17d01m36s",
            "-43d01m09s",
            "-12d05m44s",
            "-45d46m44s",
            "04d59m33s",
            "12d23m28s",
            "22d00m52s",
            "40d44m02s",
            "58d48m42s",
            "-37d12m2s",
        ]
    )

    nsources = len(names)
    nfreqs = 1

    # Stokes I fluxes.
    # Sources peeled from GLEAM come from Hurley-Walker+2017.
    # Fornax's value is from McKinley+2015.
    stokes = np.zeros((4, nfreqs, nsources)) * units.Jy
    stokes[0, :, :] = [60, 1370, 280, 390, 377, 861, 1340, 7920, 11900, 750] * units.Jy

    # Spectral indices.
    # Fluxes for sources peeled from GLEAM are given at 200 MHz.
    # Flux of Fornax was given at 154 MHz.
    reference_frequency = (
        np.array([200 for i in range(nsources - 1)] + [154]) * 1e6 * units.Hz
    )
    spectral_index = [
        -0.96,
        -0.50,
        -0.96,
        -0.99,
        -1.07,
        -0.86,
        -0.22,
        -0.78,
        -0.41,
        -0.825,
    ]

    # Make a SkyModel
    ateam_model_params = {
        "name": names,
        "ra": ra,
        "dec": dec,
        "stokes": stokes,
        "spectral_type": "spectral_index",
        "spectral_index": spectral_index,
        "reference_frequency": reference_frequency,
        "history": "Load sources brighter than 87 Jy at 154 MHz from GLEAM.\n",
    }
    ateam_model = SkyModel(**ateam_model_params)

    ateam_model.write_skyh5(ATEAM_MODEL_FILE, clobber=False)


def make_ptsrc_model(fch):
    # Load GLEAM-like and A-Team SkyModel objects, making them if they do not exist
    # in the default path
    if not GL_MODEL_FILE.exists():
        make_gleam_like_model()
    gleam_like = SkyModel.from_file(GL_MODEL_FILE)
    if not ATEAM_MODEL_FILE.exists():
        make_ateam_model()
    ateam = SkyModel.from_file(ATEAM_MODEL_FILE)

    # Evaluate the models at a given frequency
    f = np.atleast_1d(H4C_FREQS[fch])
    gleam_like_f = gleam_like.at_frequencies(
        f, freq_interp_kind="cubic", nan_handling="clip", inplace=False
    )
    ateam_f = ateam.at_frequencies(
        f, freq_interp_kind="cubic", nan_handling="clip", inplace=False
    )

    # Speactral indexes must be set to None before concatnating.
    # Otherwise, pyradiosky will complain.
    gleam_like_f.spectral_index = None
    ateam_f.spectral_index = None

    ptsrc = gleam_like_f.concat(ateam_f, inplace=False)

    utils.write_sky(ptsrc, "ptsrc", fch)


def make_confusion_map(freq: Quantity, nside: int = 256) -> Quantity:
    """Make a confusion map at a given frequency. Output NSIDE=128."""
    ref_freq = 154e6 * units.Hz

    # Load source confusion data
    data_file = f"{utils.SKYDIR}/gleam_like/gleam_like_fainter.npz"

    with np.load(data_file) as data:
        flux_ref = data["flux_density"]
        colat = data["theta"]
        lon = data["phi"]
        spectral_index = data["spec_index"]

    # Pixel indices of confusion sources
    pixel_inds = hp.ang2pix(nside, colat, lon, nest=False)

    # Power law interpolation to get source fluxes at the given frequency.
    flux_f = flux_ref * (freq / ref_freq) ** spectral_index

    # Initialize confusion map
    confusion_map = np.zeros(hp.nside2npix(nside))

    # Loop over pixel indices of the sources and add fluxes to the right pixels.
    for i, p in enumerate(pixel_inds):
        confusion_map[p] += flux_f[i]

    # Divde by pixel area to conver to Jy/sr
    confusion_map = confusion_map / hp.nside2pixarea(nside)

    # Now convert from Jy/sr to kelvin
    confusion_map = (confusion_map * units.Jy / units.sr).to(
        units.K, equivalencies=units.brightness_temperature(freq * units.Hz)
    )

    return confusion_map


def make_gsm_map(freq: Quantity, nside: int = 256, smooth=True) -> Quantity:
    """Generate a GSM map at a given frequency.

    Ouput is Galactic coordinates."""
    gsm = GlobalSkyModel(freq_unit=freq.unit)
    gsm_map = gsm.generate(freq.value)

    m_nside = hp.npix2nside(gsm_map.size)

    # Smooth with 1 deg Gaussian
    if smooth:
        gsm_map = hp.smoothing(gsm_map, fwhm=np.pi / 180, pol=False)

    # Downgrade from NSIDE=512. Default is 128, enough for HREA.
    # This preserve the flux integral which is what we need.
    # No further normalisation is required.
    if nside != m_nside:
        gsm_map = hp.ud_grade(gsm_map, nside)

    return gsm_map * units.K


def make_healpix_type_sky_model(
    hmap: Quantity,
    freq: Quantity,
    nside: int,
    inframe="galactic",
    outframe="icrs",
    to_point=True,
) -> SkyModel:
    """Construct a HEALPix-type SkyModel object given a HEALPix map"""
    npix = hp.nside2npix(nside)

    stokes = np.zeros((4, 1, npix)) * hmap.unit
    stokes[0, :, :] = hmap

    # SkyModel parameters.
    params = {
        "component_type": "healpix",
        "nside": nside,
        "hpx_inds": np.arange(npix),
        "hpx_order": "ring",
        "spectral_type": "full",
        "freq_array": freq,
        "stokes": stokes,
        "frame": inframe,
    }
    sky_model = SkyModel(**params)

    # Convert healpix to point
    if to_point:
        sky_model.healpix_to_point()

    # Transform coordinates of the GSM pixels from Galactic to ICRS.
    if outframe != inframe:
        sky_model.transform_to(outframe)

    return sky_model


def make_gsm_model(fch: int, nside: int = 256) -> SkyModel:
    """Make a GSM SkyModel at a given frequency channel."""
    freq = H4C_FREQS[fch]
    gsm_map = make_gsm_map(freq, nside=nside, smooth=True)
    gsm_model = make_healpix_type_sky_model(
        gsm_map, freq, nside, inframe="galactic", outframe="icrs", to_point=True
    )
    utils.write_sky(gsm_model, f"gsm_nside{nside}", fch)


def make_diffuse_model(fch: int, nside: int = 256) -> SkyModel:
    """Make a diffuse SkyModel (GSM + confusion at a given frequency channel."""
    freq = H4C_FREQS[fch]
    gsm_map = make_gsm_map(freq, nside=nside, smooth=True)
    confusion_map = make_confusion_map(freq, nside=nside)
    diffuse_map = gsm_map + confusion_map
    diffuse_model = make_healpix_type_sky_model(
        diffuse_map, freq, nside, inframe="galactic", outframe="icrs", to_point=True
    )
    utils.write_sky(diffuse_model, f"diffuse_nside{nside}", fch)


"""===CLI==="""

option_freq_range = click.option(
    "-fr",
    "--freq-range",
    nargs=2,
    default=(0, len(H4C_FREQS)),
    show_default=True,
    metavar="START STOP",
    type=click.IntRange(0, len(H4C_FREQS)),
    help="Frequency channel range (Pythonic)",
)
option_freqs = click.option(
    "-fch",
    "--freqs",
    multiple=True,
    default=None,
    type=click.IntRange(0, len(H4C_FREQS)),
    metavar="FREQ_CHAN",
    help="Frequency channel, allow multiple, add to --freq-range",
)
option_nside = click.option("--nside", default=256, show_default=True)


def _parse_freqs(freq_range, freqs):
    freq_chans = np.arange(*freq_range)
    if freqs:
        extra_freqs = freqs[np.isin(freqs, freq_chans, invert=True)]
        freq_chans = np.append(freq_chans, extra_freqs)
    return freq_chans


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Make SkyModel at given frequencies.

    Frequencies are based on H4C data.
    Outputs are written to the default directories, i.e. "./sky_models/<type>".
    """
    pass


@cli.command("gsm")
@option_freq_range
@option_freqs
@option_nside
def gsm(freq_range, freqs, nside):
    """Make GSM SkyModels.

    Frequencies are based on H4C data. Outputs are written to the default directories.
     i.e. "./sky_models/gsm_nside{nside}".
    """
    for fch in _parse_freqs(freq_range, freqs):
        make_gsm_model(fch, nside)


@cli.command("diffuse")
@option_freq_range
@option_freqs
@option_nside
def diffuse(freq_range, freqs, nside):
    """Make diffuse (GSM + confusion) SkyModels .

    Frequencies are based on H4C data. Outputs are written to the default directories,
     i.e. "./sky_models/diffuse{nside}".
    """
    for fch in _parse_freqs(freq_range, freqs):
        make_diffuse_model(fch, nside)


@cli.command("ptsrc")
@option_freq_range
@option_freqs
def ptsrc(freq_range, freqs):
    """Make point source SkyModels.

    Frequencies are based on H4C data. Outputs are written to the default directories,
     i.e. "./sky_models/ptsrc".
    """
    for fch in _parse_freqs(freq_range, freqs):
        make_ptsrc_model(fch)


if __name__ == "__main__":
    cli()
