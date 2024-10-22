"""
Create the covariance.h5 file containing the frequency-frequency covariance for a GRF.

This file was originally written by Zac Martinot, and was used to generate the
covariance for the H4C sims. The original filename was "do_signal_sim.py".
"""
import subprocess
import logging
import numpy as np
from astropy import cosmology
from redshifted_gaussian_fields import generator
import time
from rich.console import Console
from . import utils
from ._cli_utils import _get_sbatch_program
from .slurm import slurmify

cns = Console()
logger = logging.getLogger(__name__)

def compute_grf_covariance(test_mode: bool = False, ell_max: int = 1250):
    freqs = utils.FREQS_DICT['H6C']
    
    k0 = np.logspace(-2.,1.,11)
    a = k0**(-2.7)

    normalization_point = 0.2
    normalization_amplitude = 1.

    Pspec = generator.ParameterizedGaussianPowerSpectrum(a, k0, renormalization=(normalization_point, normalization_amplitude), term_type='flat_gauss')

    nu_axis = 1e-6*freqs # MHz
    del_nu = np.diff(nu_axis)[0]

    ell_axis = np.arange(0,ell_max+1)

    if test_mode:
        cns.print("On test mode")
        nu_axis = nu_axis[500:502]
        ell_axis = ell_axis[:64]

    cosmo = cosmology.Planck15
    Np = 15
    eps = 1e-15
    gcfg = generator.GaussianCosmologicalFieldGenerator(cosmo, Pspec, nu_axis, del_nu, ell_axis, Np=Np, eps=eps)

    cns.print("Computing covariance...")
    t1 = time.time()
    gcfg.compute_cross_frequency_angular_power_spectrum()
    t2 = time.time()
    cns.print(f"Elapsed time: {(t2 - t1)/60.} minutes.")

    save_file_path = utils.SKYDIR / "raw" / "covariance.h5"
    gcfg.save_covariance_data(save_file_path)
    cns.print(f"Saved covariance data to {save_file_path}.")

@slurmify('grf-covariance', time="2-12:00:00", defaulttasks=48, partition='RM-shared')
def run_compute_grf_covariance(
    test_mode: bool = False, ell_max: int = 1250,
):
    tm = "--test-mode" if test_mode else "--production"
    return f"time python vsim.py grf-covariance --local --ell-max {ell_max} {tm}"
    