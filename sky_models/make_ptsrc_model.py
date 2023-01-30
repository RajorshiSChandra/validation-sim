from multiprocessing import Pool
import numpy as np
from astropy.table import Table
from astropy.coordinates import Longitude, Latitude
import astropy.units as u
from pyradiosky import SkyModel
import utils

FREQ_ARRAY = utils.freqs * u.Hz

ptsrc_dir = utils.SKYDIR / 'ptsrc'

if not ptsrc_dir.exists():
    ptsrc_dir.mkdir(parents=True)

def make_gleam_like_model(data_file):
    with np.load(data_file) as data:
        flux_ref = data['flux_density']
        colat = data['theta']
        lon = data['phi']
        spectral_index = data['spec_index']

    # Put synthetic sources into a SkyModel
    nsources = len(flux_ref)

    # Names are required for point sources
    names = [f'gl{i:06d}' for i in range(nsources)]

    ra = Longitude(lon * u.rad)
    dec = Latitude(((np.pi/2) - colat) * u.rad)

    stokes = np.zeros((4, 1, nsources)) * u.Jy
    stokes[0, :, :] = flux_ref * u.Jy 

    # Reference frequency is 154 MHz
    ref_freq = 154e6 * np.ones(nsources) * u.Hz

    gl_model_params = {
        'name': names,
        'ra': ra,
        'dec': dec,
        'stokes': stokes,
        'spectral_type': 'spectral_index',
        'reference_frequency': ref_freq,
        'spectral_index': spectral_index,
        'history': 'Load gleam-like sources\n'
    }

    gl_model = SkyModel(**gl_model_params)
    return gl_model


def make_bright_gleam_model(data_file):
    gleam_df = Table.read(data_file).to_pandas()
    
    # Reindex to use "Name" as indices
    gleam_df.set_index('Name', inplace=True)
    
    # Only four sources in GLEAM are brighter than the sources in 
    # the gleam-like catalog, i.e., brither than 87 Jy at 154 MHz.
    # GLEAM doesn't actually provide flux at 154 MHz, so we selected sources
    # providing the flux cut at 151 and 158 subbands, and got these four.
    valid_sources = [b'GLEAM J215706-694117', b'GLEAM J043704+294009', 
                     b'GLEAM J122906+020251', b'GLEAM J172031-005845']
    gleam_cut = gleam_df.loc[valid_sources]

    # Get the subband fluxes
    subband_freqs = [76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 
                     166, 174, 181, 189, 197, 204, 212, 220, 227]
    flux_columns = [f'int_flux_{f:03d}' for f in subband_freqs]
    fluxes = gleam_cut[flux_columns].values.T

    # Get RA/Dec
    ra = Longitude(gleam_cut['RAJ2000'].values * u.degree)
    dec = Latitude(gleam_cut['DEJ2000'].values * u.degree)

    # Get names
    names = gleam_cut.index.astype(str).to_list()
    
    # Prepare other parameters for the SkyModel object
    nsources = len(names)
    nfreqs = len(subband_freqs)
    freq_array = np.array(subband_freqs) * 1e6 * u.Hz
    stokes = np.zeros((4, nfreqs, nsources)) * u.Jy
    stokes[0, :, :] = fluxes * u.Jy 
    
    gleam_model_params = {
        'name': names,
        'ra': ra,
        'dec': dec,
        'stokes': stokes,
        'spectral_type': 'subband',
        'freq_array': freq_array,
        'history': 'Load sources brighter than 87 Jy at 154 MHz from GLEAM.\n'
    }

    gleam_model = SkyModel(**gleam_model_params)
    return gleam_model


def make_ateam_model():
    """Make a SkyModel for sources peeled from GLEAM."""
    # The first 9 sources are peeled from GLEAM (Hurley-Walker+2017)
    names = ['3C444',
             'CentaurusA',
             'HydraA',
             'PictorA',
             'HerculesA',
             'VirgoA',
             'Crab',
             'CygnusA',
             'CassiopeiaA',
             'FornaxA'
            ]
    
    # RA/Dec. 
    # For Fornax A, represent it as a single source using its host galaxy position
    ra = Longitude(
        ['22h14m16s', 
         '13h25m28s', 
         '09h18m06s', 
         '05h19m50s',
         '16h51m08s',
         '12h30m49s',
         '05h34m32s',
         '19h59m28s',
         '23h23m28s',
         '3h22m42s'] 
    )
    dec = Latitude(
        ['-17d01m36s',
         '-43d01m09s',
         '-12d05m44s',
         '-45d46m44s',
         '04d59m33s',
         '12d23m28s',
         '22d00m52s',
         '40d44m02s',
         '58d48m42s',
         '-37d12m2s'
        ]
    )
    
    nsources = len(names)
    nfreqs = 1
    
    # Stokes I fluxes. 
    # Sources peeled from GLEAM come from Hurley-Walker+2017.
    # Fornax's value is from McKinley+2015.
    stokes = np.zeros((4, nfreqs, nsources)) * u.Jy
    stokes[0, :, :] = [60, 1370, 280, 390, 377, 
                       861, 1340, 7920, 11900, 750] * u.Jy
    
    # Spectral indices. 
    # Fluxes for sources peeled from GLEAM are given at 200 MHz.
    # Flux of Fornax was given at 154 MHz. 
    reference_frequency = np.array(
        [200 for i in range(nsources-1)] + [154]
    ) * 1e6 * u.Hz
    spectral_index = [-0.96, -0.50, -0.96, -0.99, -1.07, 
                      -0.86, -0.22, -0.78, -0.41, -0.825]
    
    # Make a SkyModel
    ateam_model_params = {
        'name': names,
        'ra': ra,
        'dec': dec,
        'stokes': stokes,
        'spectral_type': 'spectral_index',
        'spectral_index': spectral_index,
        'reference_frequency': reference_frequency,
        'history': 'Load sources brighter than 87 Jy at 154 MHz from GLEAM.\n'
    }
    ateam_model = SkyModel(**ateam_model_params)
    
    return ateam_model


gleam_like_data_file = f'{utils.SKYDIR}/gleam_like/gleam_like_brighter.npz'
gleam_like = make_gleam_like_model(gleam_like_data_file)

#gleam_data_file = f'{utils.SKYDIR}/GLEAM_EGC_v2.fits'
#bright_gleam = make_bright_gleam_model(gleam_data_file)

ateam = make_ateam_model()


def concat_models(i):
    f = np.atleast_1d(FREQ_ARRAY[i])
 #   bright_gleam_f = bright_gleam.at_frequencies(
 #       f, freq_interp_kind='cubic',
 #       nan_handling='clip', inplace=False
 #   )
    gleam_like_f = gleam_like.at_frequencies(
        f, freq_interp_kind='cubic',
        nan_handling='clip', inplace=False
    )
    ateam_f = ateam.at_frequencies(
        f, freq_interp_kind='cubic',
        nan_handling='clip', inplace=False
    )
    
 #   bright_gleam_f.spectral_index = None
    gleam_like_f.spectral_index = None
    ateam_f.spectral_index = None
    
    ptsrc = gleam_like_f.concat(ateam_f, inplace=False)
    # ptsrc.concat(ateam_f, inplace=True)

    utils.write(ptsrc, 'ptsrc', i)

if __name__ == "__main__":
    with Pool() as p:
        p.map(concat_models, range(FREQ_ARRAY.size))
