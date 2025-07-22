"""Functions for computing expected EOR power spectra.

These functions were originally utilities in Zac's notebooks for inspecting the EOR
simulation outputs. We put them here for simpler re-useability.
"""
import hera_pspec as hp
import numpy as np
from redshifted_gaussian_fields import ParameterizedGaussianPowerSpectrum, con as zconst
from functools import partial
from pathlib import Path
import h5py
from astropy import cosmology

def get_pspec_from_covariance_file(full_file_path: str | Path) -> ParameterizedGaussianPowerSpectrum:

    with h5py.File(full_file_path, 'r') as h5f:
        iparams = h5f['input_parameters']
        cosmology_name = iparams['cosmology_name'][()].decode()
        cosmo = getattr(cosmology, cosmology_name)

        nu_axis = iparams['nu_axis'][()]
        del_nu = iparams['del_nu'][()]
        ell_axis = iparams['ell_axis'][()]

        a = iparams['a'][()]
        k0 = iparams['k0'][()]
        renorm0 = iparams['renormalization_point'][()]
        renorm1 = iparams['renormalization_amplitude'][()]
        renormalization = (renorm0, renorm1)
        term_type = iparams['term_type'][()].decode()

        return ParameterizedGaussianPowerSpectrum(a, k0, renormalization, term_type)

def get_planck15_cosmo():
    
    H0 = 67.74
    h = H0/100.

    return hp.conversions.Cosmo_Conversions(
        Om_L=0.6911, Om_b = .02230/h**2, Om_c = 0.1188/h**2, H0=H0
    )

def true_power_spectrum(
    cosmo: hp.conversions.Cosmo_Conversions,
    pgps: ParameterizedGaussianPowerSpectrum, 
    k: np.ndarray, 
    z: float, 
    rescale_idx: float=-11, 
    littleh: bool=True
) -> np.ndarray:
    """Compute the true power spectrum from a GRF model."""
    h = cosmo.h if littleh else 1
    
    Pk = pgps(h * k)
    
    nu_z = cosmo.z2f(z)
    
    # redshift dimming factor
    Pk *= (zconst.nu_e * 1e6/nu_z)**-2
    Pk *= (nu_z / 100e6)**rescale_idx
    
    # little-h "units"
    Pk *= h**3
    
    # K^2 to mK^2
    Pk *= 1e6
    
    return Pk

def get_freqs(uvp: hp.UVPSpec, spw: int) -> np.ndarray:
    """Obtain the frequencies within a spectral window in a UVPSpec object."""
    return uvp.freq_array[uvp.spw_freq_array == spw]

def get_zs(cosmo: hp.conversions.Cosmo_Conversions, uvp: hp.UVPSpec, spw: int) -> np.ndarray:
    """Obtain the redshifts in a UVPSpec object within a given spw."""
    spw_freqs = get_freqs(uvp, spw)
    return cosmo.f2z(spw_freqs)

def get_ks(cosmo: hp.conversions.Cosmo_Conversions, uvp: hp.UVPSpec, spw: int, littleh: bool=True) -> tuple[float, float]:
    """Get the k_s critical scale."""
    zs = get_zs(cosmo, uvp, spw)
    dr = np.diff(np.array([cosmo.DC(z, little_h=littleh) for z in zs]))
    
    return np.pi * np.mean(1/np.abs(dr)), np.mean(zs)

def tapered_power_spectrum(
    cosmo: hp.conversions.Cosmo_Conversions, 
    pgps: ParameterizedGaussianPowerSpectrum, 
    k: np.ndarray, 
    uvp: hp.UVPSpec, 
    spw: int, 
    rescale_idx: float=-11, 
    littleh: bool=True
) -> np.ndarray:
    z = np.mean(get_zs(cosmo, uvp, spw))
    Pk = true_power_spectrum(cosmo, pgps, k, z, rescale_idx=rescale_idx, littleh=littleh)

    freqs = get_freqs(uvp, spw)
    alpha = cosmo.dRpara_df(z)
    delta_nu = np.median(np.diff(freqs))
    
    x = 0.5 * k * alpha * delta_nu
    taper = np.sinc(x/np.pi)**2
    
    return Pk * taper

def expected_power_spectrum(
    cosmo: hp.conversions.Cosmo_Conversions, 
    pgps: ParameterizedGaussianPowerSpectrum, 
    k: np.ndarray, 
    uvp: hp.UVPSpec, 
    spw: int, 
    rescale_idx: float=-11, 
    littleh: bool=True, 
    nterms: int=20
) -> np.ndarray:
    ks, z = get_ks(cosmo, uvp, spw)
    
    tps = partial(tapered_power_spectrum, cosmo=cosmo, pgps=pgps, spw=spw, uvp=uvp, rescale_idx=rescale_idx, littleh=littleh)
    
    Pk = tps(k=k)
    for n in range(1,nterms):
        Pk += tps(k=k + 2*n*ks) + tps(k=k - 2*n*ks)
    
    return Pk