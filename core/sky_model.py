"""Utilities for creating sky models."""
import logging
import subprocess
from functools import cache, cached_property

import h5py
import healpy as hp
import numpy as np
from astropy import units
from astropy.coordinates import Latitude, Longitude
from astropy.units import Quantity
from pygdsm import GlobalSkyModel
from pyradiosky import SkyModel
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import gaussian_kde

from . import utils
from ._cli_utils import _get_sbatch_program

logger = logging.getLogger(__name__)

H4C_FREQS = utils.FREQS_DICT["H4C"] * units.Hz
ATEAM_MODEL_FILE = utils.RAWSKYDIR / "ateam.skyh5"


def write_sky(sky: SkyModel, model: str, channel: int):
    """Write a particular channel of a sky model to file."""
    d = utils.SKYDIR / model
    d.mkdir(parents=True, exist_ok=True)
    sky.write_skyh5(f"{d}/fch{channel:04d}.skyh5", clobber=True)


def randsphere(n, theta_range=(0, np.pi), phi_range=(0, 2 * np.pi)):
    """
    Generate random angular theta, phi points on the sphere.

    Parameters
    ----------
    n: integer
        The number of randoms to generate

    Returns
    -------
    theta,phi: tuple of arrays
    """
    phi = np.random.random(n)
    phi = phi * (phi_range[1] - phi_range[0]) + phi_range[0]

    cos_theta_min = np.cos(theta_range[0])
    cos_theta_max = np.cos(theta_range[1])

    v = np.random.random(n)
    v *= cos_theta_max - cos_theta_min
    v += cos_theta_min

    theta = np.arccos(v)

    return theta, phi


def dnds_franzen(s, a=None, norm=False):
    """
    Compute the dn/dS function from Franzen 2019.

    S here is in Jy.
    """
    if a is None:
        a = [3.52, 0.307, -0.388, -0.0404, 0.0351, 0.00600]  # from franzen 2019
    out = 10 ** (sum(np.log10(s) ** i * aa for i, aa in enumerate(a)))
    if norm:
        return out
    else:
        return s**-2.5 * out


def make_grf_eor_model(model_file: str, channels: list[int], label: str = ""):
    """Make a GRF EoR SkyModel.

    The model file is here assumed to contain Nfreqs healpix maps. The format of the
    file is custom, as defined by @zacharymartinot, and the sky map is created by
    his redshifted_gaussian_field code.
    """
    model_dir = utils.SKYDIR / "eor"

    with h5py.File(model_dir / model_file, "r") as fl:
        nside = int(fl["nside"][()])
        freqs = fl["frequencies_mhz"][:] * 1e6 << units.Hz  # units Hz

        # HEALPix array -- dimension (nfreqs, npix) -- unit Jy/sr
        # We must read in the whole thing to set the monopole.
        hmaps = fl["healpix_maps_Jy_per_sr"][:, :]

    hmaps <<= units.Jy / units.sr

    npix = hp.nside2npix(nside)

    # Shift the maps values so there are no negative values
    # Note: we move the WHOLE sky at ALL frequencies up by a set amount so that the
    # minimum value is positive
    # We do not move parts of the map by different amounts.
    hmaps -= hmaps.min()

    # Initialize stokes array
    stokes = np.zeros((4, 1, npix)) * units.Jy / units.sr

    # Setup SkyModel params
    params = {
        "component_type": "healpix",
        "nside": nside,
        "hpx_order": "ring",
        "hpx_inds": np.arange(npix),
        "spectral_type": "full",
        "freq_array": [freqs[0]] * units.Hz,
        "stokes": stokes,
        "frame": "icrs",
    }

    eor_model = SkyModel(**params)

    outdir = utils.SKYDIR / f"eor-grf-{nside}{label}"
    outdir.mkdir(parents=True, exist_ok=True)

    for fch in channels:
        logger.info(f"Constructing channel {fch}")
        eor_model.stokes[0, 0] = hmaps[fch]
        eor_model.freq_array[0] = freqs[fch]

        eor_model.write_skyh5(outdir / f"fch{fch:04d}.skyh5", clobber=True)


class FranzenSourceCounts:
    """Class for dealing with source counts from Franzen et al. (2019)."""

    def __init__(self, smin, smax, base_svec=None, base_cdf=None, npoints: int = 10000):
        self.smin = smin
        self.smax = smax

        if base_svec is None:
            self._base_svec = np.logspace(4, -10, npoints)
        else:
            self._base_svec = base_svec

        if base_cdf is None:
            self._base_cdf = np.zeros(len(self._base_svec))
            for i, x in enumerate(self._base_svec[1:], start=1):
                self._base_cdf[i] = (
                    integrate.quad(dnds_franzen, x, self._base_svec[i - 1])[0]
                    + self._base_cdf[i - 1]
                )
        else:
            self._base_cdf = base_cdf

        self._mask = (self._base_svec > self.smin) & (self._base_svec < self.smax)
        self.svec = self._base_svec[self._mask]

        self._cdf = self._base_cdf.copy()
        self._cdf = self._cdf[self._mask]
        self._cdf -= self._cdf.min()

    def with_bounds(
        self, smin: float | None = None, smax: float | None = None
    ) -> "FranzenSourceCounts":
        """Return a new object with different Smin/Smax."""
        if smin is None:
            smin = self.smin
        if smax is None:
            smax = self.smax

        return FranzenSourceCounts(smin, smax, self._base_svec, self._base_cdf)

    def with_nsources_per_pixel(
        self, nsources: int, nside: int
    ) -> "FranzenSourceCounts":
        """Return a new object that sets Smin based on a desired number of sources."""
        pixarea = hp.nside2pixarea(nside)
        smin_idx = np.argwhere(self.cumulative_source_density * pixarea > nsources)[0][
            0
        ]
        return FranzenSourceCounts(
            smin=self.svec[smin_idx],
            smax=self.smax,
            base_cdf=self._base_cdf,
            base_svec=self._base_svec,
        )

    @cached_property
    def cumulative_source_density(self):
        """The CDF of dn/ds."""
        return self._cdf

    @cached_property
    def normalized_cdfvec(self):
        """The normalzed CDF (maximum density of 1)."""
        return self.cumulative_source_density / self.nbar

    @cached_property
    def nbar(self) -> float:
        """The mean density of galaxies (per steradian) within the given flux range."""
        return self.cumulative_source_density.max()

    def sample_source_count(self, solid_angle: float) -> int:
        """Get a random number specifying the number of sources in a given region."""
        return np.random.poisson(solid_angle * self.nbar)

    @cached_property
    def invcdf(self):
        """The inverse CDF."""
        return InterpolatedUnivariateSpline(self.normalized_cdfvec, self.svec)

    def sample_fluxes(self, solid_angle: float) -> np.ndarray:
        """Sample from dn/ds in a given solid angle."""
        n = self.sample_source_count(solid_angle)
        return self.invcdf(np.random.uniform(size=n))

    def sample_pixel_flux(self, nside: int, n: int = 1) -> np.ndarray:
        """Generate samples of the total flux in a pixel."""
        solid_angle = hp.nside2pixarea(nside)
        out = np.zeros(n)
        for i in range(n):
            out[i] = np.sum(self.sample_fluxes(solid_angle))
        return out

    def pixel_flux_distribution(self, nside: int, n: int = 10000):
        """Get a sample-able distribution of flux within pixels.

        This does a semi-brute-force calculation, finding the distribution by
        Monte-Carloing many pixels of total flux (drawing point sources in each
        pixel from the source count distribution). Using an analytic formula is not
        possible. Even using the mean and variance (which is analytically tractable)
        is not useful since the distribution is highly non-Gaussian in general.
        """
        fluxes = self.sample_pixel_flux(nside, n=n)
        return gaussian_kde(np.log10(fluxes))


@cache
def franzen_base() -> FranzenSourceCounts:
    """Get a base source counts object that enables caching the bigger computations."""
    return FranzenSourceCounts(smin=1e-10, smax=1e4)


def make_gleam_like_model(
    max_flux_density: float = 100.0,
    seed: int = 42,
    mean_spectral_index: float = 0.8,
    sigma_spectral_index: float = 0.05,
    nside: int = 256,
) -> SkyModel:
    """
    Create a set of point sources that have the same statistical properties as GLEAM.

    This creates a set of point sources with a flux distribution consistent with that
    measured by Franzen et al. (2019) using GLEAM measurements. The brightest sources
    can be omitted with the `max_flux_density` parameter, to allow for putting real
    bright sources on top.

    The distribution of sources is assumed to be uniform on the sky, which is not
    hugely inconsistent with reality given that the bulk of sources are faint and
    therefore unlikely to be highly biased.

    The number of sources returned depends on the `nside` parameter, which sets the
    assumed resolution of the telescope. We aim for one source per healpix pixel on
    average. If we had more sources, they would be "confused" and appear as a background
    thermal-noise-like rumble in the diffuse map.

    Parameters
    ----------
    max_flux_density: float, optional
        The maximum flux density of the sources. The default is 100.0.
    seed: int, optional
        The random seed to use. The default is 42.
    mean_spectral_index: float, optional
        The mean spectral index of the sources. The default is 0.8.
    sigma_spectral_index: float, optional
        The standard deviation of the spectral index of the sources. The default is 0.05.
    nside: int, optional
        The healpix nside parameter to use, to choose the approximate number of
        sources. Fainter sources are not simulated. The default is 256.

    Returns
    -------
    gl_model
        A SkyModel representing the GLEAM-like sources.

    """
    np.random.seed(seed)
    source_counts = (
        franzen_base()
        .with_bounds(smax=max_flux_density)
        .with_nsources_per_pixel(nsources=1, nside=nside)
    )
    flux_ref = source_counts.sample_fluxes(solid_angle=4 * np.pi)
    nsources = len(flux_ref)
    colat, lon = randsphere(nsources)

    spectral_index = np.random.normal(
        loc=mean_spectral_index, scale=sigma_spectral_index, size=nsources
    )

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
        "frame": "icrs",
    }

    return SkyModel(**gl_model_params)


def make_ateam_model() -> SkyModel:
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
        "frame": "icrs",
    }
    return SkyModel(**ateam_model_params)


def make_ptsrc_model(channels: list[int], nside: int = 256, **kw):
    """Create a point-source model."""
    # Load GLEAM-like and A-Team SkyModel objects, making them if they do not exist
    # in the default path
    if not ATEAM_MODEL_FILE.exists():
        ateam = make_ateam_model()
        ateam.write_skyh5(ATEAM_MODEL_FILE)
    else:
        ateam = SkyModel.from_file(ATEAM_MODEL_FILE)
        ateam.write_skyh5(
            ATEAM_MODEL_FILE, clobber=True
        )  # This keeps it updated for newer versions of pyradiosky

    gleam_like = make_gleam_like_model(
        nside=nside, max_flux_density=ateam.stokes[0].min().to_value("Jy"), **kw
    )

    # Evaluate the models at a given frequency
    for fch in channels:
        gleam_like_f = gleam_like.at_frequencies(
            H4C_FREQS[fch : (fch + 1)],
            freq_interp_kind="cubic",
            nan_handling="clip",
            inplace=False,
        )
        ateam_f = ateam.at_frequencies(
            H4C_FREQS[fch : (fch + 1)],
            freq_interp_kind="cubic",
            nan_handling="clip",
            inplace=False,
        )

        # Speactral indexes must be set to None before concatnating.
        # Otherwise, pyradiosky will complain.
        gleam_like_f.spectral_index = None
        ateam_f.spectral_index = None

        ptsrc = gleam_like_f.concat(ateam_f, inplace=False)

        write_sky(ptsrc, f"ptsrc{nside}", fch)


def make_confusion_map(
    freq: Quantity,
    nside: int = 256,
    smax=0.03,
    mean_spectral_index: float = 0.8,
) -> Quantity:
    """Make a confusion map at a given frequency. Output NSIDE=128."""
    source_counts = franzen_base().with_bounds(smin=0, smax=smax)
    kde = source_counts.pixel_flux_distribution(nside=nside, n=10000)

    fluxes = 10 ** kde.resample(size=hp.nside2npix(nside))
    ref_freq = 154e6 * units.Hz

    # The spectral indices of each pixel come from a distribution defined by
    # (nu / nu_ref)**-gamma, where gamma is normally distributed. According to
    # https://stats.stackexchange.com/a/335990/81338 and ignoring the second-order
    # terms, this has a mean of the original mean of gamma and variance proportional
    # to the mean -- i.e. we can simply scale all the pixels down by the mean spectral
    # index and we'll get the right mean and variance (to first order).
    fluxes *= (freq / ref_freq) ** -mean_spectral_index

    # Divde by pixel area to conver to Jy/sr
    confusion_map = fluxes / hp.nside2pixarea(nside)

    # Now convert from Jy/sr to kelvin
    confusion_map = (confusion_map * units.Jy / units.sr).to(
        units.K, equivalencies=units.brightness_temperature(freq * units.Hz)
    )

    return confusion_map


def make_gsm_map(
    freq: Quantity, nside: int = 256, smooth=True, gsm: GlobalSkyModel | None = None
) -> Quantity:
    """
    Generate a GSM map at a given frequency.

    Ouput is Galactic coordinates.
    """
    if gsm is None:
        gsm = GlobalSkyModel(freq_unit=freq.unit)
    else:
        assert gsm.freq_unit == freq.unit

    gsm_map = gsm.generate(freq.value)

    m_nside = hp.npix2nside(gsm_map.size)

    # Smooth with 1 deg Gaussian
    if smooth:
        gsm_map = hp.smoothing(gsm_map, fwhm=np.pi / 180, pol=False)

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
    """Construct a HEALPix-type SkyModel object given a HEALPix map."""
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


def make_gsm_model(channels: list[int], nside: int = 256) -> SkyModel:
    """Make a GSM SkyModel at a given frequency channel."""
    freqs = H4C_FREQS[channels]
    gsm = GlobalSkyModel(freq_unit=freqs[0].unit)

    for fch, freq in zip(channels, freqs):
        gsm_map = make_gsm_map(freq, nside=nside, smooth=True, gsm=gsm)
        gsm_model = make_healpix_type_sky_model(
            gsm_map, freq, nside, inframe="galactic", outframe="icrs", to_point=True
        )
        write_sky(gsm_model, f"gsm_nside{nside}", fch)


def make_diffuse_model(channels: int, nside: int = 256) -> SkyModel:
    """Make a diffuse SkyModel (GSM + confusion at a given frequency channel."""
    freqs = H4C_FREQS[channels]
    gsm = GlobalSkyModel(freq_unit=freqs[0].unit)

    for fch, freq in zip(channels, freqs):
        gsm_map = make_gsm_map(freq, nside=nside, smooth=True, gsm=gsm)
        confusion_map = make_confusion_map(freq, nside=nside)
        diffuse_map = gsm_map + confusion_map
        diffuse_model = make_healpix_type_sky_model(
            diffuse_map, freq, nside, inframe="galactic", outframe="icrs", to_point=True
        )
        write_sky(diffuse_model, f"diffuse_nside{nside}", fch)


def run_make_sky_model(
    sky_model: str,
    channels: list[int],
    nside: int,
    slurm_override: tuple[tuple[str, str]],
    skip_existing: bool,
    dry_run: bool,
    split_freqs: bool = False,
    label: str = "",
):
    """Run the sky model creation via SLURM."""
    model = f"{sky_model}{nside}"
    out_dir = utils.SKYDIR / f"{model}"
    logdir = utils.LOGDIR / f"skymodel/{model}"

    logdir.mkdir(parents=True, exist_ok=True)

    # We want to override the job-name to be <sky_model>-<fch>-<ch>, but the last two
    # variables have to be accessed in the loop, so we will be instead override it to
    # a Python string formatting pattern and format it in the loop.
    # Note that click default `slurm_overrride` to (), and we want it to be "2D" tuple
    slurm_override = slurm_override + (
        ("job-name", "{sky_model}-fch{fch:04d}" if split_freqs else sky_model),
        (
            "output",
            "{logdir}/fch{fch:04d}_%J.out" if split_freqs else "{logdir}/%J.out",
        ),
    )

    if "time" not in [x[0] for x in slurm_override]:
        slurm_override = slurm_override + (("time", "0-00:15:00"),)

    # Make the SBATCH script minus hera-sim-vis.py command
    program = _get_sbatch_program(gpu=False, slurm_override=slurm_override)

    sbatch_dir = utils.REPODIR / "batch_scripts/skymodel"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    if split_freqs:
        for fch in channels:
            logger.info(f"Working on frequency channel {fch}")

            outfile = out_dir / f"fch{fch:04d}.skyh5"

            # Check if output file already existed, if clobber is False
            if skip_existing and outfile.exists():
                logger.warning(f"File {outfile} exists, skipping")
                continue

            cmd = f"time python vsim.py sky-model {sky_model} --local --nside {nside} --freq-range {fch} {fch+1} --label '{label}'"

            if utils.HPC_CONFIG["slurm"]:
                # Write job script and submit
                sbatch_file = sbatch_dir / f"{sky_model}_fch{fch:04d}.sbatch"

                logger.info(f"Creating sbatch file: {sbatch_file}")
                # Now, join the job script with the hera-sim-vis.py command
                # and format the job-name
                job_script = "\n".join([program, "", cmd, ""]).format(
                    sky_model=sky_model, fch=fch, logdir=logdir
                )
                with open(sbatch_file, "w") as fl:
                    fl.write(job_script)

                if not dry_run:
                    subprocess.call(f"sbatch {sbatch_file}".split())

                logger.debug(f"\n===Job Script===\n{job_script}\n===END===\n")
            else:
                logger.info(f"Running the simulation locally\nCommand: {cmd}")
                if not dry_run:
                    subprocess.call(cmd.split())
    else:
        channels = sorted(channels)
        groups = [[channels[0]]]
        for ch in channels[1:]:
            if ch == groups[-1][-1] + 1:
                groups[-1].append(ch)
            else:
                groups.append([ch])

        chan_opt = ""
        for g in groups:
            if len(g) == 1:
                chan_opt += f"--channels {g[0]} "
            else:
                chan_opt += f"--channels {g[0]}~{g[-1]+1}"

        cmd = f"time python vsim.py sky-model {sky_model} --local --nside {nside} --eor-label '{label}' {chan_opt}"

        if utils.HPC_CONFIG["slurm"]:
            # Write job script and submit
            sbatch_file = sbatch_dir / f"{sky_model}_allfreqs.sbatch"

            logger.info(f"Creating sbatch file: {sbatch_file}")
            job_script = "\n".join([program, "", cmd, ""]).format(
                sky_model=sky_model, logdir=logdir
            )
            with open(sbatch_file, "w") as fl:
                fl.write(job_script)

            if not dry_run:
                subprocess.call(f"sbatch {sbatch_file}".split())

            logger.debug(f"\n===Job Script===\n{job_script}\n===END===\n")
        else:
            logger.info(f"Running the simulation locally\nCommand: {cmd}")
            if not dry_run:
                subprocess.call(cmd.split())
