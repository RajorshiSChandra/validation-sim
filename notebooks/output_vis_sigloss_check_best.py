#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
output_vis_sigloss_check.py
===========================

Signal-loss check for the HERA H6C validation simulations.

This script estimates the delay-spectrum power spectrum of a redundant baseline
group along TWO paths and is designed so the two can be compared to quantify the
signal loss incurred by coherent (visibility-level) averaging:

  1. COHERENT path  (potentially lossy):
       redundant baselines are averaged together at the *visibility* level
       (complex average) and a single power spectrum is formed from the average.

  2. INCOHERENT path (reference, ~loss-free):
       a power spectrum is formed for *each* baseline first, and the resulting
       *power spectra* are averaged together (with autos providing the noise
       covariance model).

Pipeline overview (top to bottom):
  IMPORTS  ->  PARAMETERS  ->  HELPER FUNCTIONS  ->
  load pre-concatenated cross/auto UVH5  ->  select redundant group  ->
  define spectral window(s)  ->  NaN->0 cleaning  ->  Nsample spectral
  smoothing  ->  excise low-Nsample baselines  ->  redundant visibility
  average  ->  COHERENT power spectrum (uvp)  ->  combine cross+autos  ->
  INCOHERENT power spectrum (uvpinc)  ->  NaN->0 on stats  ->  incoherent
  blpair average (uvpspec_averaged)  ->  save both products to PSpecContainer.

Everything a user normally edits (paths, sky/beam model, channel/chunk ranges,
baseline selection, polarization, toggles) lives in the PARAMETERS section.

Converted from the Jupyter notebook of the same name; dead/commented-out code
paths were removed, parameters hoisted to the top, and section documentation
added. The analysis logic itself is unchanged.

This is the merged "best" build: it keeps the full configuration catalogs
and all diagnostics (from the verbose conversion) and adds parameterized
pspec kwargs, memory-management helpers (free_named_objects / inplace
controls), fail-fast file checks, and beam/range guards (from the condensed
conversion). Defaults reproduce the notebook's numerical results exactly.
"""

# ===========================================================================
# 1. IMPORTS & GLOBAL SETUP
# ===========================================================================

####################################################################################################################################################################################################################################################################################
# ________  _________ ___________ _____              
# |_   _|  \/  || ___ \  _  | ___ \_   _|             
#   | | | .  . || |_/ / | | | |_/ / | |               
#   | | | |\/| ||  __/| | | |    /  | |               
#  _| |_| |  | || |   \ \_/ / |\ \  | |               
#  \___/\_|  |_/\_|    \___/\_| \_| \_/               
                                                    
                                                    
# ______  ___  _____  _   __  ___  _____  _____ _____ 
# | ___ \/ _ \/  __ \| | / / / _ \|  __ \|  ___/  ___|
# | |_/ / /_\ \ /  \/| |/ / / /_\ \ |  \/| |__ \ `--. 
# |  __/|  _  | |    |    \ |  _  | | __ |  __| `--. \
# | |   | | | | \__/\| |\  \| | | | |_\ \| |___/\__/ /
# \_|   \_| |_/\____/\_| \_/\_| |_/\____/\____/\____/ 
#                                                    
##############################################################################################################################

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'   # required for HDF5 reads on lustre

import gc
import sys
import glob
import re
import copy
import ctypes
import itertools
import argparse
import subprocess
from copy import deepcopy
from collections import OrderedDict as odict

import numpy as np
import h5py
import hdf5plugin            # REQUIRED so the HDF5 compression plugins are available
import healpy as hlp
import pandas as pd
pd.set_option('display.max_rows', 1000)
import psutil
from pympler import asizeof

from scipy import stats
from astropy import constants
from scipy import constants, interpolate   # NOTE: rebinds `constants` to scipy (matches notebook)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Radio-astronomy / HERA stack
# import hera_sim
import hera_pspec as hp
from hera_pspec import pspecdata
from hera_pspec import utils as pspec_utils
from hera_pspec import uvpspec_utils as uvputils
import hera_cal as hc
from hera_cal import io, utils, redcal, apply_cal, datacontainer, abscal
from hera_qm import ant_metrics, ant_class, xrfi
from hera_filters import dspec
from uvtools.plot import plot_antpos, plot_antclass
import linsolve

from pyuvdata import UVData, UVCal, UVBeam
from pyuvdata import utils as uvutils

from IPython.display import display, HTML   # notebook-only; harmless in a script
# display(HTML("<style>.container { width:100% !important; }</style>"))
# %config InlineBackend.figure_format = 'retina'
# _ = np.seterr(all='ignore')  # get rid of red warnings

# Matplotlib defaults (match the notebook look)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = "18"

# Report the versions of the key packages (also defines the module-level
# `__version__`, which the custom average_spectra() below stamps into history).
for repo in ['numpy', 'scipy', 'astropy', 'hera_cal', 'hera_qm', 'hera_filters', 'pyuvdata', 'hera_pspec']:
    exec(f'from {repo} import __version__')
    print(f'{repo}: {__version__}')

# ===========================================================================
# 2. PARAMETERS  --  EVERYTHING A USER NORMALLY EDITS LIVES HERE
# ===========================================================================
#
# Pick the sky/beam model by (un)commenting the appropriate MODEL_DIR below, set
# the channel/chunk ranges, the target redundant baseline (length + angle), the
# polarization, the beam file, and the output toggles.  Nothing below this
# section should normally need editing.

from pathlib import Path
# ------------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------------

BASE_OUTDIR= Path("/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/outputs")
# BASE_OUTDIR= Path("/lustre/aoc/projects/hera/kmandar/repos/validation-sim/outputs")  
# / "ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyprb"
# / "ptsrc256/nt17280-00288chunks-HERA_custom_subset_cba81417555edaffd87557575713cb61.txt-nonred"

# PTSRC SKY
# sky_type = "ptsrc"

# Airy
# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# ideal ENU, diameter Airy var (deltaD ~ <.2m)        airyprb, idealT
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyprb")
# real ENU, diameter Airy var (deltaD ~ <.2m)         airyprb, idealF
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyprb")
# ideal ENU, tilt Airy var (deltZa ~ 2-3 degree)      airytilt, idealT
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")
# real ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealF
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")

# Vivaldi
# ideal ENU, ideal Vivaldi       vivaldired, idealT
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
# real ENU, ideal Vivaldi       vivaldired, idealF
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")


# EOR SKY
sky_type = "eor_ns" 

# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")

# ideal ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealT
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")
# real ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealF
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")

# Vivaldi
# ideal ENU, ideal Vivaldi       vivaldired, idealT
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
# real ENU, ideal Vivaldi       vivaldired, idealF
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")

# Gaussian###################
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")




# EOR SKY PLAYGROUND =============================================

# Gaussian###################
# MODEL_DIR  = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix_freqclone.03001951._beammapperant_airyred-nonred_airyred")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix_spatmean_pwlw/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_80MHzref_gauss_nonspectral_skysd111_eoroffsetfix.29493475._beammapperant_airyred-nonred_airyred/")

    # Rlzn 222
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_80MHzref_gauss_nonspectral_skysd111_eoroffsetfix.29493475._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_80MHzref_gauss_nonspectral_skysd111_eoroffsetfix.29493475._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")

# Airy ######################
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_skysd111_eoroffsetfix.5cfe1b35._beammapperant_airyred-nonred_airyred/")
    # Rlzn 222
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_skysd111_eoroffsetfix.5cfe1b35._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_tilt_skysd111_eoroffsetfix.06127f13._beammapperant_airytilt-nonred_airytilt/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_skysd111_eoroffsetfix.5cfe1b35._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_nonspectral_80MHzref_eoroffsetfix.a74a7b9d._beammapperant_airyred-nonred_airyred/")
    # Rlzn 556
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_nonspectral_80MHzref_eoroffsetfix.a74a7b9d._beammapperant_airyred-nonred_airyred/")
    # Rlzn 556, 14m Airy
# Rlzn 556, 14m Airy (previous active; kept in menu):
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")
# Rlzn 222, 14m Airy, fftvis cross-check (CURRENT notebook active):
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/fftvis_xcheck/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")    
    # Rlzn 556, 14m Airy, CV rlzn challenge
# MODEL_DIR  = Path("eor-grf-256/seed709_freqslic_middle_fch0273ref/fftvis_xcheck/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")
MODEL_DIR  = Path("eor-grf-256/seed909_freqslic_middle_fch0273ref/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")
    # Rlzn 556 1 tilt
# MODEL_DIR   = Path("eor-grf-256/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_1tilt_nonspectral_80MHzref_eoroffsetfix.378b3c2a._beammapperant_airytilt-nonred_airytilt/")        

# HERA Stripe Zenith Pixel ######################
# MODEL_DIR  = Path("eor-grf-256/zenith_bright_point/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_nonspectral_80MHzref_eoroffsetfix.a74a7b9d._beammapperant_airyred-nonred_airyred/")

# HERA Stripe Zenith 5 Pixel ######################
# MODEL_DIR  = Path("eor-grf-256/zenith_5_point/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_nonspectral_80MHzref_eoroffsetfix.a74a7b9d._beammapperant_airyred-nonred_airyred/")

# Isotropic ######################  
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_iso_skysd111_eoroffsetfix.920cf40b._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_iso_skysd111_eoroffsetfix.920cf40b._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_iso_skysd111_eoroffsetfix.920cf40b._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_iso_skysd111_eoroffsetfix.920cf40b._beammapperant_airyred-nonred_airyred/")



# EOR Noisy =============================================

# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR   = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# MODEL_DIR   = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-2x/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-correct-beam/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-D7m-airy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy.5cfe1b35._beammapperant_airyred-nonred_airyred/")      # seed 777
# MODEL_DIR   = Path("eor-grf-256-noisy-D7m-airy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_skysd111.5cfe1b35._beammapperant_airyred-nonred_airyred/")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR   = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
     # Rlzn 556
# MODEL_DIR   = Path("eor-grf-256-noisy-D7m-airy/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_nonspectral_80MHzref_eoroffsetfix.a74a7b9d._beammapperant_airyred-nonred_airyred/")
    # Rlzn 556 1 tilt
# MODEL_DIR   = Path("eor-grf-256-noisy-D7m-airy/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_1tilt_nonspectral_80MHzref_eoroffsetfix.378b3c2a._beammapperant_airytilt-nonred_airytilt/")        

# PTSRC Noisy =============================================

# PURE Noise =============================================

# MODEL_DIR   = Path("noise-only-300k/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")

DATA_PATH = BASE_OUTDIR / MODEL_DIR




run_batch = 250925 # args.run_batch
total_cores = 10 #args.total_cores
batch_number = 5 # args.batch

fch_min, fch_max = 271, 276          # inclusive channel range
chunk_min, chunk_max = 0, 288         # inclusive chunk range

bl_len = 29.0                        # target baseline length [m]
bl_ang = 0.0                         # target angle [deg], e.g. EW group

len_tol = 2.0                       # ± length tolerance [m]
ang_tol = 5.0                        # ± angle tolerance [deg]

pol_in = "xx"                        # one of {"xx", "yy", "xy", "yx"}

# Polarization pair fed to the power spectrum estimator (derived from pol_in).
# polpair_in = ('pI', 'pI')   # alternative: pseudo-Stokes I
polpair_in = (pol_in, pol_in)

# ---------------------------------------------------------------------------
# Primary beam (used to normalize / convert visibilities Jy -> mK)
# ---------------------------------------------------------------------------
beam_path = '/home/herastore02-1/HERA_Validation_rchandra/'
# Alternative beam_path options:
# beam_path = '/lustre/aoc/projects/hera/rchandra/Raj_UVH5_PSPECH5_Rlzn_NO_FRF_TAVG_INTRLV_Files_3/Beams/'
# beam_path = '/lustre/aoc/projects/hera/h6c-analysis/IDR2/beams/'

# Beam file -- uncomment the one matching the sky/beam model chosen in MODEL_DIR.
# To get correct polarization info, use "like with like" (match beam pols to data).
    # Vivaldi:
# beamfile = os.path.join(beam_path, 'NF_HERA_Vivaldi_efield_beam_healpix.fits')
# beamfile = os.path.join(beam_path, 'NF_HERA_Vivaldi_efield_beam_extrap_0.fits')
# beamfile = os.path.join(beam_path, 'NF_HERA_Vivaldi_efield_beam_healpix_pstokes.fits')   # pStokes pI
    # Airy (chromatic):
# beamfile = os.path.join(beam_path, 'airy_beam_7.0m_decay_0.3dBdeg_start_70.0deg_healpix.fits')
    # Airy (achromatic, frequency-constant, ref 80 MHz):
# beamfile = os.path.join(beam_path, 'airy_beam_7.0m_freqconst_ref80MHz_decay_0.3dBdeg_start_70.0deg_healpix.fits')   # 7 m dish
beamfile = os.path.join(beam_path, 'airy_beam_14.0m_freqconst_ref80MHz_decay_0.3dBdeg_start_70.0deg_healpix.fits')    # 14 m dish  (ACTIVE)
    # Gaussian:
# beamfile = os.path.join(beam_path, 'gaussian_beam_7.0m_decay_0.3dBdeg_start_70.0deg_healpix.fits')
    # Gaussian (achromatic):
# beamfile = os.path.join(beam_path, 'gaussian_beam_7.0m_freqconst_ref80MHz_decay_0.3dBdeg_start_70.0deg_healpix.fits')
    # Isotropic:
# beamfile = os.path.join(beam_path, 'isotropic_beam_fullsphere_healpix.fits')

# ---------------------------------------------------------------------------
# Pre-processing toggles
# ---------------------------------------------------------------------------
cut_switch = 1              # 1 -> excise baselines/pols with low Nsamples, 0 -> keep all
nsample_excise_thresh = 0   # keep keys whose max(Nsamples) is strictly greater than this

# ---------------------------------------------------------------------------
# Output naming / scratch locations
# ---------------------------------------------------------------------------
batchnum = 0               # batch index used in output filenames (distinct from batch_number above)

# Scratch directory for the temporary "Junk" pspec files written by pspec_run()
# (these are deleted again later in the NaN-cleaning section).
save_path_H4C_SIM_NOISE_Chunked_RedGrp_NO_RedBlAvg_TAVG_FRF_PSPECH5_pI = "/home/herastore02-1/H6C_scratch_rchandra/Sig_Loss_1_pI"

# Processing-stage labels embedded in the final output filenames.
#   proc1[0]="cutbl"  -> baselines were cut;   proc1[1]="allbl"  -> all baselines kept
#   proc2[0]="cutlst" -> LSTs were cut;        proc2[1]="alllst" -> all LSTs kept
proc1 = ["cutbl", "allbl"]
proc2 = ["cutlst", "alllst"]

# ---------------------------------------------------------------------------
# POWER-SPECTRUM ESTIMATOR PARAMETERS  (defaults match the notebook exactly)
# ---------------------------------------------------------------------------
PSPEC_INPUT_DATA_WEIGHT          = "identity"        # pspec input_data_weight
PSPEC_NORM                       = "I"               # pspec norm
PSPEC_TAPER                      = "blackman-harris" # delay-taper window
PSPEC_XANT_FLAG_THRESH           = 0.95              # pspec_run xant_flag_thresh
STORE_INCOHERENT_COVARIANCE      = True              # incoherent: store_cov
STORE_INCOHERENT_COVARIANCE_DIAG = True              # incoherent: store_cov_diag
INCOHERENT_COV_MODEL             = "autos"           # incoherent: cov_model
INCOHERENT_AVERAGE_ERROR_FIELD   = "autos_diag"      # average_spectra error_field (notebook value)
INCOHERENT_AVERAGE_TIME          = False             # average_spectra time_avg
SPW_RANGES                       = None              # override spw_ranges; None -> [(0, fch_max-fch_min)]

# ---------------------------------------------------------------------------
# MEMORY / INPLACE CONTROLS  (defaults reproduce the notebook's results)
# ---------------------------------------------------------------------------
FREE_INTERMEDIATE_OBJECTS = True    # del large UVData/PSpecData objects once consumed
CLEAN_UVPINC_INPLACE      = True    # NaN->0 on uvpinc in place (numerically identical)
AVERAGE_SPECTRA_INPLACE   = False   # False = exact notebook behaviour; True = memory-saving (safe)

# ---------------------------------------------------------------------------
# SANITY CHECKS on the user parameters above
# ---------------------------------------------------------------------------
assert fch_min <= fch_max, f"fch_min ({fch_min}) must be <= fch_max ({fch_max})"
assert chunk_min <= chunk_max, f"chunk_min ({chunk_min}) must be <= chunk_max ({chunk_max})"

# ---------------------------------------------------------------------------
# OPTIONAL: file-completeness check (independent diagnostic, runs at the end)
# ---------------------------------------------------------------------------
mfc_directory = Path(
    "../outputs/eor-grf-256/zenith_5_point/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_nonspectral_80MHzref_eoroffsetfix.a74a7b9d._beammapperant_airyred-nonred_airyred"
)
mfc_channels = range(227, 316)   # upper bound exclusive
mfc_chunks   = range(0, 96)      # upper bound exclusive

# ===========================================================================
# 3. HELPER FUNCTIONS  (general-purpose utilities used throughout)
# ===========================================================================

# Better memory release back to the OS on Linux (call after freeing big arrays).
def malloc_trim():
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def show_ram():
    """Print current resident-set-size (RAM) of this process."""
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss
    print(f"Notebook RAM usage: {rss/1e9:.2f} GB")


def free_named_objects(*names):
    """Drop large top-level objects once downstream products are already made.

    Gated by FREE_INTERMEDIATE_OBJECTS; deletes the named globals, then runs
    gc + malloc_trim and reports RAM. Names not present are silently skipped.
    """
    if not FREE_INTERMEDIATE_OBJECTS:
        return
    removed = []
    for name in names:
        if name in globals():
            del globals()[name]
            removed.append(name)
    if removed:
        print("Freed large objects:", ", ".join(removed))
        gc.collect()
        malloc_trim()
        show_ram()


# ------------------------------------------------------------------
# HELPER: build filename for a given (channel, chunk)
# ------------------------------------------------------------------

def fname_for(ch, chunk):
    """
    Return full path to fch####_chunk#####.uvh5 for given channel, chunk.
    """
    return DATA_PATH / f"fch{ch:04d}_chunk{chunk:05d}.uvh5"

# ASSORTED HELPER FUNCTIONS
####################################################################################################################################################################################################################################################################################
#   ___   _____ _____  ___________ _____ ___________      
#  / _ \ /  ___/  ___||  _  | ___ \_   _|  ___|  _  \     
# / /_\ \\ `--.\ `--. | | | | |_/ / | | | |__ | | | |     
# |  _  | `--. \`--. \| | | |    /  | | |  __|| | | |     
# | | | |/\__/ /\__/ /\ \_/ / |\ \  | | | |___| |/ /      
# \_| |_/\____/\____/  \___/\_| \_| \_/ \____/|___/       
                                                        
                                                        
#  _   _  _____ _     ______ ___________                  
# | | | ||  ___| |    | ___ \  ___| ___ \                 
# | |_| || |__ | |    | |_/ / |__ | |_/ /                 
# |  _  ||  __|| |    |  __/|  __||    /                  
# | | | || |___| |____| |   | |___| |\ \                  
# \_| |_/\____/\_____/\_|   \____/\_| \_|                 
                                                        
                                                        
# ______ _   _ _   _ _____ _____ _____ _____ _   _  _____ 
# |  ___| | | | \ | /  __ \_   _|_   _|  _  | \ | |/  ___|
# | |_  | | | |  \| | /  \/ | |   | | | | | |  \| |\ `--. 
# |  _| | | | | . ` | |     | |   | | | | | | . ` | `--. \
# | |   | |_| | |\  | \__/\ | |  _| |_\ \_/ / |\  |/\__/ /
# \_|    \___/\_| \_/\____/ \_/  \___/ \___/\_| \_/\____/ 
#                                                         
##############################################################################################################################
                                                        


def zero_fraction_nsample(uvd):
    """
    Compute the number and fraction of zero entries in uvd.nsample_array.

    Parameters
    ----------
    uvd : object
        An object with attribute `nsample_array`, assumed to be a NumPy array.

    Returns
    -------
    zeros : int
        Count of entries equal to zero.
    total : int
        Total number of entries in the array.
    fraction : float
        Fraction of zeros (zeros/total).
    """
    arr = np.asarray(uvd.nsample_array)
    total = arr.size
    zeros = int((arr == 0).sum())
    fraction = zeros / total if total else 0.0
    return zeros, total, fraction

##############################################################################################################################

def summarise_uvd_quality(uvd):
    """
    Summarise the fraction of NaNs in data_array, nsample_array,
    and the fraction of flagged entries in the flags array of a UVData-like object.

    Prints:
        data_array : x/y NaNs (p%)
        nsample_array : x/y NaNs (p%)
        flags_array : x/y flagged (p%)
    """
    summary = {}

    # Data array NaNs
    if hasattr(uvd, 'data_array'):
        arr = uvd.data_array
        nans = np.isnan(arr).sum()
        total = arr.size
        summary['data_array'] = (nans, total)

    # nsample_array NaNs
    if hasattr(uvd, 'nsample_array'):
        arr = uvd.nsample_array
        nans = np.isnan(arr).sum()
        total = arr.size
        summary['nsample_array'] = (nans, total)

    # flags array (boolean) -> flagged count
    flag_attr = None
    for attr in ('flag_array', 'flags', 'flags_array'):
        if hasattr(uvd, attr):
            flag_attr = attr
            break
    if flag_attr:
        arr = getattr(uvd, flag_attr)
        flagged = np.count_nonzero(arr)
        total_flags = arr.size
        summary[flag_attr] = (flagged, total_flags)

    # Print results
    for name, (count, total) in summary.items():
        pct = 100 * count / total if total else 0
        if 'flag' in name:
            print(f"{name}: {count}/{total} flagged ({pct:.2f}%)")
        else:
            print(f"{name}: {count}/{total} NaNs ({pct:.2f}%)")

    return summary

##############################################################################################################################

def summarise_nan_stats(uvp, max_stats=3, max_spws=2, max_indices=5):
    """
    Print a truncated summary of NaN positions in uvp.stats_array,
    plus an overall count of NaNs vs total data points.

    Parameters
    ----------
    uvp : UVPSpec
        The UVPSpec object to inspect.
    max_stats : int
        Maximum number of statistic fields to report.
    max_spws : int
        Maximum number of spectral windows (per stat) to report.
    max_indices : int
        Maximum number of NaN indices to show per (stat, spw).
    """
    if not hasattr(uvp, 'stats_array'):
        print("No stats_array in this UVPSpec.")
        return

    # Overall NaN summary
    total_nans = 0
    total_points = 0
    for stat, spw_dict in uvp.stats_array.items():
        for arr in spw_dict.values():
            total_nans += np.isnan(arr).sum()
            total_points += arr.size
    pct = (100 * total_nans / total_points) if total_points > 0 else 0
    print(f"Overall NaNs in stats_array: {total_nans} / {total_points} ({pct:.2f}%)\n")

    stats_reported = 0
    for stat, spw_dict in uvp.stats_array.items():
        if stats_reported >= max_stats:
            remaining = len(uvp.stats_array) - max_stats
            print(f"...and {remaining} more stats fields.")
            break
        # count spws that actually have NaNs
        spws_with_nans = {spw: arr for spw, arr in spw_dict.items() if np.isnan(arr).any()}
        if not spws_with_nans:
            stats_reported += 1
            continue

        print(f"Stat '{stat}' has NaNs in {len(spws_with_nans)} spw(s):")
        spws_reported = 0
        for spw, arr in spws_with_nans.items():
            if spws_reported >= max_spws:
                remaining = len(spws_with_nans) - max_spws
                print(f"  ...and {remaining} more spws for '{stat}'.")
                break
            nan_coords = np.argwhere(np.isnan(arr))
            count_nans = nan_coords.shape[0]
            print(f"  spw {spw}: {count_nans} NaNs; showing up to {max_indices}:")
            for coord in nan_coords[:max_indices]:
                print(f"    index {tuple(coord)}")
            if count_nans > max_indices:
                print(f"    ...and {count_nans - max_indices} more indices.")
            spws_reported += 1

        stats_reported += 1

##############################################################################################################################

def stats_nan_fraction_per_blpair(uvp):
    """
    Compute the percentage of NaNs across all stats_array entries for each baseline-pair.

    Parameters
    ----------
    uvp : UVPSpec
        A UVPSpec object with a stats_array attribute.

    Returns
    -------
    dict
        Mapping each blpair (int) to its fraction of NaNs in stats_array (as a percentage).
    """
    if not hasattr(uvp, 'stats_array'):
        return {}

    blpairs = np.array(uvp.blpair_array)
    nan_counts = np.zeros(len(blpairs), dtype=int)
    total_counts = np.zeros(len(blpairs), dtype=int)

    # Iterate over every stat field and every spectral window
    for stat, spw_dict in uvp.stats_array.items():
        for arr in spw_dict.values():
            arr = np.asarray(arr)
            # assume first axis corresponds to baseline-pairs
            n_blp = arr.shape[0]
            # flatten other dims
            flattened = arr.reshape(n_blp, -1)
            nan_counts += np.isnan(flattened).sum(axis=1)
            total_counts += flattened.shape[1]

    # Compute percentage
    percent_nan = 100 * nan_counts / total_counts

    return dict(zip(blpairs.tolist(), percent_nan.tolist()))

##############################################################################################################################

def data_nan_fraction_per_blpair(uvp):
    """
    Compute the percentage of NaNs across uvp.data_array entries for each baseline-pair.

    Parameters
    ----------
    uvp : UVPSpec
        A UVPSpec object with a data_array attribute.

    Returns
    -------
    dict
        Mapping each blpair ID to its fraction of NaNs in data_array (as a percentage).
    """
    if not hasattr(uvp, 'data_array'):
        return {}

    blpairs = np.array(uvp.blpair_array)
    nan_counts = np.zeros(len(blpairs), dtype=int)
    total_counts = np.zeros(len(blpairs), dtype=int)

    for spw, arr in uvp.data_array.items():
        arr = np.asarray(arr)
        # Ensure baseline-pair axis is first
        if arr.shape[0] != len(blpairs):
            # find any axis matching blpair count and move it to front
            match_axes = [i for i, s in enumerate(arr.shape) if s == len(blpairs)]
            if match_axes:
                arr = np.moveaxis(arr, match_axes[0], 0)
        n_blp = arr.shape[0]
        # flatten all but baseline-pair axis
        flattened = arr.reshape(n_blp, -1)
        nan_counts += np.isnan(flattened).sum(axis=1)
        total_counts += flattened.shape[1]

    percent_nan = 100 * nan_counts / total_counts
    return dict(zip(blpairs.tolist(), percent_nan.tolist()))


def check_missing_files(directory, channels, chunks):
    """Report any fch####_chunk#####.uvh5 files missing from `directory`.

    Independent diagnostic: scans the (channel x chunk) grid and prints which
    expected files are absent. Returns the list of missing Paths.
    """
    missing_files = []
    for ch in channels:
        for chunk in chunks:
            file_path = directory / f"fch{ch:04d}_chunk{chunk:05d}.uvh5"
            if not file_path.exists():
                missing_files.append(file_path)

    total_expected = len(channels) * len(chunks)
    print(f"Checked {total_expected} expected files")

    if missing_files:
        print(f"Missing {len(missing_files)} files:")
        for path in missing_files:
            print(path)
    else:
        print("All files exist.")
    return missing_files

# ===========================================================================
# 4. LOAD DATA  --  metadata, redundant group, pre-concatenated cross/autos
# ===========================================================================
#
# The heavy per-(channel, chunk) read + concatenation is assumed to have been
# done already (it produced uvd_cross_*.uvh5 and uvd_autos_*.uvh5). Here we just
# (a) read one file as a metadata template, (b) find the redundant baseline
# group nearest (bl_len, bl_ang), and (c) read the concatenated cross & auto
# UVData objects from disk.
# ------------------------------------------------------------------
# STEP 1: Build a metadata UVData to define redundant group
# ------------------------------------------------------------------

# Use the first available (fch, chunk) as a metadata template
ref_ch = fch_min
ref_ck = chunk_min
ref_fn = fname_for(ref_ch, ref_ck)
if not ref_fn.exists():
    raise FileNotFoundError(f"Reference file {ref_fn} not found.")

uvd_meta = UVData()
# read only metadata (faster)
uvd_meta.read_uvh5(ref_fn, read_data=False)
print(f"Loaded metadata from: {ref_fn}")
print("Nants_data:", uvd_meta.Nants_data, "Nbls:", uvd_meta.Nbls)

# ------------------------------------------------------------------
# STEP 2: Find redundant group near (bl_len, bl_ang)
# ------------------------------------------------------------------

red_bls, lens, angs = hp.utils.get_reds(
    uvd_meta,
    bl_error_tol=2.0,
    add_autos=True,                  # autos included in grouping
    bl_len_range=(10.0, 100.0),      # broad range
    bl_deg_range=(0.0, 180.0),
    pick_data_ants=True,
)

print("Found", len(red_bls), "redundant groups.")

# choose group whose (length, angle) is closest to (bl_len, bl_ang)
best_idx = None
best_score = None
for i, (L, A) in enumerate(zip(lens, angs)):
    if (abs(L - bl_len) <= len_tol) and (abs(A - bl_ang) <= ang_tol):
        score = (L - bl_len) ** 2 + (A - bl_ang) ** 2
        if (best_score is None) or (score < best_score):
            best_score = score
            best_idx = i

if best_idx is None:
    raise RuntimeError(
        f"No redundant group found within "
        f"{len_tol} m and {ang_tol} deg of (L={bl_len}, A={bl_ang})."
    )

print(f"Selected redundant group index {best_idx}: "
      f"L = {lens[best_idx]:.2f} m, A = {angs[best_idx]:.2f}°")

redgrp = red_bls[best_idx]   # list of (ant1, ant2) for this group
print("Number of baselines in group (incl. autos if present):", len(redgrp))
print("redgrp:", redgrp)

# Cross-only version (unpolarized)
redgrp_unpol = [(a1, a2) for (a1, a2) in redgrp if a1 != a2]
print("Cross-only baselines in group:", len(redgrp_unpol))

# Autos associated with this group
autos_set = sorted({(a, a) for (a1, a2) in redgrp for a in (a1, a2)})
print("Associated autos:", autos_set)

# ------------------------------------------------------------------
# Assumes these are already defined above:
#   MODEL_DIR, DATA_PATH, fch_min, fch_max, chunk_min, chunk_max
# ------------------------------------------------------------------

# Rebuild tags from MODEL_DIR (same logic as before)
base = MODEL_DIR.name
path = MODEL_DIR.parent
# e.g. "nt17280-00288chunks-HERA_custom_subset_idealF_cba814...txt-nonred_airytilt"
print("MODEL_DIR base:", base)
print("MODEL_DIR path:", path)

# m_sky_type = re.search(r"([^_]+)", path)
# sky_type_tag = m_sky_type.group(1) if m_sky_type else "sky"

m_ideal = re.search(r"subset_([^_]+)_", base)
ideal_tag = m_ideal.group(1) if m_ideal else "ideal"

m_airy = re.search(r"nonred_([^_]+)$", base)
airy_tag = m_airy.group(1) if m_airy else "airy"

# print("sky_type_tag:", sky_type_tag)
print("ideal_tag:", ideal_tag)
print("airy_tag :", airy_tag)

# sky_type = sky_type_tag

# Build filename stem with channel / chunk limits
tag_stem = (
    f"{ideal_tag}_{airy_tag}"
    f"_fch{fch_min:04d}-{fch_max:04d}"
    f"_ck{chunk_min:05d}-{chunk_max:05d}"
    f"_bl{bl_len:.1f}m"
)

cross_outfile = DATA_PATH / f"uvd_cross_{tag_stem}.uvh5"
autos_outfile = DATA_PATH / f"uvd_autos_{tag_stem}.uvh5"

print("Expected cross file:", cross_outfile)
print("Expected autos file:", autos_outfile)

# ------------------------------------------------------------------
# Check existence and read in
# ------------------------------------------------------------------
if cross_outfile.exists() and autos_outfile.exists():
    print("\nFound both concatenated files. Reading them in...")

    uvd_combined = UVData()
    uvd_combined.read_uvh5(cross_outfile)

    uvd_autos = UVData()
    uvd_autos.read_uvh5(autos_outfile)

    print("\n=== Loaded CROSS data ===")
    print("Nbls:", uvd_combined.Nbls, "Ntimes:", uvd_combined.Ntimes, "Nfreqs:", uvd_combined.Nfreqs)
    print("Unique antpairs (cross):", uvd_combined.get_antpairs())
    print("Polarizations:", uvd_combined.polarization_array)

    print("\n=== Loaded AUTOS data ===")
    print("Nbls:", uvd_autos.Nbls, "Ntimes:", uvd_autos.Ntimes, "Nfreqs:", uvd_autos.Nfreqs)
    print("Unique antpairs (autos):", uvd_autos.get_antpairs())
    print("Polarizations:", uvd_autos.polarization_array)

else:
    missing = []
    if not cross_outfile.exists():
        missing.append(str(cross_outfile))
    if not autos_outfile.exists():
        missing.append(str(autos_outfile))
    raise FileNotFoundError(
        "Concatenated power-spectrum input file(s) missing -- create them "
        "first (run the concatenation step):\n  " + "\n  ".join(missing)
    )

# ===========================================================================
# 5. SELECT REDUNDANT BASELINE GROUP  (build redgrp_unpol_comb / autos_set)
# ===========================================================================
#
# Re-derives the redundant groups from the metadata object and isolates the
# group matching (bl_len, bl_ang). `redgrp_unpol_comb` is the list of (ant1,ant2)
# cross baselines in that group that are actually present in the data.
####################################################################################################################################################################################################################################################################################
# Select specific redundant baseline group

# Calc all red bls groups in the data
red_bls, lens, angs = hp.utils.get_reds(
    uvd_meta,
    bl_error_tol=1.0,
    add_autos=False,
    pick_data_ants=True,
)

print("List of All Redundant Baseline Groups:")
print("-----------------------------------------------------")

for length, angle, bl_group in zip(lens, angs, red_bls):
    num_baselines = len(bl_group)
    print(f"{length:.2f} m, {angle:.2f}°, {num_baselines} baselines")

print("-----------------------------------------------------")


### Choose baselines

pol = ['ee','nn']

# Get redundant baselines contained in this object
red_bls, lens, angs = hp.utils.get_reds(
    uvd_meta,
    bl_error_tol=1.0,
    add_autos=True,
    bl_len_range=(10.0, 100.0),
    bl_deg_range=(0.0, 180.0),
    pick_data_ants=True,
)
for i in range(0, len(red_bls)):
    print("red_bls, ", red_bls[i])
print("lens ", lens)
print("angs ", angs)
print("                                   ")

# Initialize selection parameters
selection_index = []  # Store the index of selected baselines
# bl_len = bl_len  # Target baseline length in meters
# bl_ang = bl_ang  # We want only the 0 degree group

print("bl_len, bl_ang ",bl_len, bl_ang)

################################################################################################
print("################################################################################################")

# Select baselines close to the specified length (28m) and angle (0°)
for val, (ilen, iang) in enumerate(zip(lens, angs)):
    if np.abs(ilen - bl_len) <= 5 and np.abs(iang - bl_ang) <= 5:
        selection_index = val  # Store the index of the selected group
        break

print("selection_index", selection_index)
print("                                   ")

################################################################################################
print("################################################################################################")

# Retrieve the selected redundant baseline group for 0° angle
redgrp0 = red_bls[int(selection_index)]
print("Selected 0° redundant group:", redgrp0)
print("                                   ")

################################################################################################
print("################################################################################################")

# Find the intersection of selected baselines with data
red_bls_dat = uvd_meta.get_antpairs()
red_bls_int = [x for x in redgrp0 if x in red_bls_dat]

print("Intersection with available data:", red_bls_int)
print("                                   ")

################################################################################################
print("################################################################################################")

# Add polarization information for the selected baselines
redgrp_pol = [ii + (jj,) for ii in red_bls_int for jj in pol]
redgrp_unpol = [ii for ii in red_bls_int]

print("red_bls_int len", len(red_bls_int))
print("                                   ")
print("red_bls_int", red_bls_int)
print("                                   ")
print("redgrp_pol len", len(redgrp_pol))
print("                                   ")
print("redgrp_pol", redgrp_pol)
print("                                   ")
print("c0", redgrp_pol[0:2])
print("                                   ")

# Combine the results into single lists for propagation
redgrp_pol_comb = list(itertools.chain.from_iterable([redgrp_pol]))
print("redgrp_pol_comb", redgrp_pol_comb)
print("                                   ")

redgrp_unpol_comb = list(itertools.chain.from_iterable([redgrp_unpol]))
print("redgrp_unpol_comb", redgrp_unpol_comb)
print("                                   ")

# ===========================================================================
# 6. SPECTRAL WINDOW(S)
# ===========================================================================
#
# A single spectral window spanning the full requested channel range. The window
# is expressed relative to fch_min, i.e. (0, fch_max - fch_min), because the
# concatenated UVData starts at channel fch_min.
spw_ranges = SPW_RANGES if SPW_RANGES is not None else [(0, fch_max - fch_min)]
print("spw_ranges ", spw_ranges)

# ===========================================================================
# 7. NaN -> 0 CLEANING  (visibility data)
# ===========================================================================
#
# Inpainted / flagged regions can carry NaNs. The power spectrum estimator
# handles missing data through flags, so any residual NaNs in the data_array
# must be zeroed first (NaN * weight = NaN would otherwise poison the FFT).
####################################################################################################################################################################################################################################################################################
# If no unflagged NaNs, set all NaNs to 0 :

# __Check for baselines with no NaNs at all (Very few baselines)__

# __Set all NaNs to 0__

# __Check for baselines with no NaNs at all (All baselines)__

summary = summarise_uvd_quality(uvd_combined)
print(summary)
summary = summarise_uvd_quality(uvd_autos)
print(summary)

from copy import deepcopy

def replace_nan_with_zero(uvd):
    uvd_no_nan = deepcopy(uvd)
    
    # Replace NaN values in the visibility data array with 0
#     for bl in redgrp_unpol_comb:
#         print(uvd_cleaned_inp.get_data(bl))
    print("1",uvd_no_nan.data_array.shape)
    uvd_no_nan.data_array = np.nan_to_num(uvd_no_nan.data_array, nan=0)
    print("2",uvd_no_nan.data_array.shape)
    
    return uvd_no_nan

####################################################################################################################################################################################################################################################################################
# Convert all NaNs to 0s; very important regardless of data 

# Sum
uvd_cleaned_inp = replace_nan_with_zero(uvd_combined)

# Autos
uvd_cleaned_inp_auto = replace_nan_with_zero(uvd_autos)

summary = summarise_uvd_quality(uvd_cleaned_inp)
print(summary)
summary = summarise_uvd_quality(uvd_cleaned_inp_auto)
print(summary)

print(redgrp_unpol_comb)

# ===========================================================================
# 8. NSAMPLE SPECTRAL SMOOTHING
# ===========================================================================
#
# Inpainted channels are recorded with Nsamples = 0, which would create gaps in
# the Nsamples spectrum and bias the downstream redundant / incoherent averages
# (and the autos-based noise estimate). Here Nsamples is made spectrally smooth
# per integration by replacing each spectral window with its mean Nsamples:
#
#     Nsamples(f) -> < Nsamples >_f      (mean over the spectral window)
#
# `view_nsamples` is a read-only inspection helper; `replace_nsamples_with_median`
# does the actual smoothing (method=1, the mean, is the active choice).
####################################################################################################################################################################################################################################################################################
# Coherent Average 

# The def coherently_average_visibilities(dc_list, flags, nsamples, rng, nint=20): used in Full Day Systematics Inspection notebook is for coherent time average

# Make Nsamples spectrally smooth to avoid post inpainting 0s in Stevendata

# __Important that Nsamples be spectrally smooth before going to red_avg__

def view_nsamples(uvd):

    #print("shp ", uvd.nsample_array.shape)
    #print("shp ", uvd.data_array[:,0,0])
    # Iterate over all baselines and polarizations
    for bl in uvd.get_antpairs():
        for pol in uvd.get_pols():
            # Get the polarization index
            pol_idx = uvd.get_pols().index(pol)
            
            # Get the nsample array for this baseline and polarization
            key = bl + (pol,)
            nsamples = uvd.get_nsamples(key)  # Shape: (Ntimes, Nfreqs)
            flags = uvd.get_flags(key)  # Shape: (Ntimes, Nfreqs)
            # np.set_printoptions(threshold=np.inf)
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            
            # Make a writable copy of the nsample array
            nsamples = np.array(nsamples, copy=True)
            
            for index, spw_range in enumerate(spw_ranges):
                print(spw_ranges[index][0], spw_ranges[index][1])
                if (pol == 'ee' and index==1) :
                    # index=0
                    tslice=0
                    print("nsamples bl spw ", bl, index, '\n',  nsamples[tslice,spw_ranges[index][0]:spw_ranges[index][1]])
                    #print("nsamples bl spw ", bl, index,  nsamples[tslice,spw_ranges[index+1][0]:spw_ranges[index+1][1]])
                    print("flags spw ", index,  flags[tslice,spw_ranges[index][0]:spw_ranges[index][1]])
                    #print("flags spw ", index,  flags[tslice,spw_ranges[index+1][0]:spw_ranges[index+1][1]])

                    # Process each time slice
                    # for t in range(nsamples.shape[0]):
                    t=tslice
                    # Calculate the median of non-zero values
                    print(nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]] > 0)
                    temp_nsamples = nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]]
                    non_zero_nsamples = temp_nsamples[nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]] > 0]
                    print("non_zero_nsamples ", non_zero_nsamples)
                    if len(non_zero_nsamples) > 0:  # Avoid empty slices
                        median_value = np.mean(non_zero_nsamples)
                    else:
                        # If all values are zero, set the median to zero
                        median_value = 0
                    print("median_value ", median_value)

#                 # Replace zeros with the median value
# #                 nsamples[t, nsamples[t] == 0] = median_value
#                 nsamples[t, :] = median_value

#             # Update the nsample_array for this baseline and polarization
            bl_idx = uvd.antpair2ind(bl)  # Get the baseline index
            print("shp ", uvd.data_array[bl_idx,0,0])
            print("shp ", uvd.nsample_array[bl_idx,2,0])
#             uvd.nsample_array[bl_idx, :, pol_idx] = nsamples

    return uvd


# view_nsamples(uvd_cleaned_inp_coher)
view_nsamples(uvd_cleaned_inp)
# print("wfgwerf")
# view_nsamples(uvd_combined)

def replace_nsamples_with_median(uvd):

    uvd_cleaned_inp_nsmp = deepcopy(uvd)
    
    # Iterate over all baselines and polarizations
    for bl in uvd.get_antpairs():
        for pol in uvd.get_pols():
            # Get the polarization index
            pol_idx = uvd.get_pols().index(pol)
            
            # Get the nsample array for this baseline and polarization
            key = bl + (pol,)
            nsamples = uvd.get_nsamples(key)  # Shape: (Ntimes, Nfreqs)

            # Make a writable copy of the nsample array
            nsamples = np.array(nsamples, copy=True)
            
            for index, spw_range in enumerate(spw_ranges):
                print(spw_ranges[index][0], spw_ranges[index][1])

                # Process each time slice
                for t in range(nsamples.shape[0]):
                    method = 1
                    
                    # MEAN #########################################################################################################################################################
                    if(method == 1):
                        # Calculate the median/mean of non-zero values
                        temp_nsamples = nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]]
                        non_zero_nsamples = temp_nsamples[nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]] > 0]
                        # print("non_zero_nsamples ", non_zero_nsamples)
                        if len(non_zero_nsamples) > 0:  # Avoid empty slices
                            # median_value = 1 #np.median(non_zero_nsamples)
                            # mean_value = np.mean(non_zero_nsamples)            # Don't include 0 values in the nsample array
                            mean_value = np.mean(temp_nsamples)            # Include 0 values in the nsample array
                            # print("int of mean_value - ", int(mean_value) )
                        else:
                            # If all values are zero, set the median to zero
                            # median_value = 0
                            mean_value = 0
                            print("else mean_value ", mean_value)

                        # Replace zeros with the median/mean value
                        # nsamples[t, nsamples[t] == 0] = median_value
                        nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]] = (mean_value) #can use int(mean_value) too # NOTE : INT Seems to change things a lot during redundant averaging and np.median(nsampple) in var_4m_autos!
                        # print("nsamples shp ", nsamples.shape)
                        
                        
                    # MEAN 2 #########################################################################################################################################################
                    if(method == 11):
                        def per_band_avg_nsamples(nsamples, band_slices):
                            '''Create new datacontainer where nsamples has been averaged per-integration in band_slices.
                            This is an approximate way to account for the fact that inpainted data is considered Nsamples=0.'''
                            out_nsamples = copy.deepcopy(nsamples)
                            for bl in out_nsamples:
                                for band in band_slices:
                                    for i in range(out_nsamples[bl].shape[0]):
                                        out_nsamples[bl][i, band] = np.mean(out_nsamples[bl][i, band])
                            return out_nsamples
                        out_nsamples = per_band_avg_nsamples(nsamples, band_slices)
                
                    
                    # NGHBR MEAN #########################################################################################################################################################
                    if(method == 2):
                        # Neighbor mean per nsample per freq per time per bls
                        temp_nsamples = nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]]
                        temp_nsamples_edg = nsamples[t,spw_ranges[index][0]-1:spw_ranges[index][1]+1]
                        zero_index = np.where(temp_nsamples == 0)
                        print("____nsamples ", t, index, pol, bl, nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]])
#                         print("check ", temp_nsamples[zero_index])
#                         print("zero_index ", zero_index[0])
#                         print("zero_index ", zero_index[0]+1)
#                         print(" test 1 ", temp_nsamples_edg[zero_index[0]+1+1] )
#                         print(" test 2 ", temp_nsamples_edg[zero_index[0]+1-1] )
                        temp_nsamples[zero_index] = [ ( temp_nsamples_edg[zero_index[0]+1+1] + temp_nsamples_edg[zero_index[0]+1-1] )/ 2 ]
                        nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]] = temp_nsamples.astype(int)
                        print("+ nsamples ", nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]])
                        
                    # 1s #########################################################################################################################################################
                    if(method == 3):
                        temp_nsamples = nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]]
                        non_zero_nsamples = temp_nsamples[nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]] > 0]
                        if len(non_zero_nsamples) > 0:  # Avoid empty slices
                            mean_value = 11 #np.mean(non_zero_nsamples)
                        else:
                            mean_value = 0
                        nsamples[t,spw_ranges[index][0]:spw_ranges[index][1]] = int(mean_value)
                
                turnon = 1
                if(turnon == 1):
                    # Update the nsample_array for this baseline and polarization
                    bl_idx = uvd.antpair2ind(bl)  # Get the baseline index
                    print("bl_idx ", bl_idx)
                    uvd_cleaned_inp_nsmp.nsample_array[bl_idx, spw_ranges[index][0]:spw_ranges[index][1], pol_idx] = nsamples[:,spw_ranges[index][0]:spw_ranges[index][1]]

    return uvd_cleaned_inp_nsmp


# uvd_cleaned_inp_nsmp = replace_nsamples_with_median(uvd_cleaned_inp_coher)
uvd_cleaned_inp_nsmp = replace_nsamples_with_median(uvd_cleaned_inp)
view_nsamples(uvd_cleaned_inp_nsmp)

# uvd_cleaned_inp = deepcopy(uvd_cleaned_inp_nsmp)
# uvd_cleaned_inp = deepcopy(uvd_cleaned_inp_coher)
# view_nsamples(uvd_cleaned_inp)

# Autos
uvd_cleaned_inp_nsmp_auto = replace_nsamples_with_median(uvd_cleaned_inp_auto)
# uvd_cleaned_inp_auto = deepcopy(uvd_cleaned_inp_nsmp_auto)


summary = summarise_uvd_quality(uvd_cleaned_inp_nsmp)
print(summary)
summary = summarise_uvd_quality(uvd_cleaned_inp_nsmp_auto)
print(summary)

# ===========================================================================
# 9. EXCISE LOW-NSAMPLE BASELINES / POLARIZATIONS
# ===========================================================================
#
# Drop (ant1, ant2, pol) keys whose max(Nsamples) <= nsample_excise_thresh, i.e.
# baselines that are effectively fully flagged. Autos are kept only for antennas
# whose polarization still survives among the cross baselines. Controlled by the
# `cut_switch` / `nsample_excise_thresh` parameters.
####################################################################################################################################################################################################################################################################################
# ____________ _________________ _____ _____  _____ _____ _____ _____ _   _ _____ 
# | ___ \ ___ \  ___| ___ \ ___ \  _  /  __ \|  ___/  ___/  ___|_   _| \ | |  __ \
# | |_/ / |_/ / |__ | |_/ / |_/ / | | | /  \/| |__ \ `--.\ `--.  | | |  \| | |  \/
# |  __/|    /|  __||  __/|    /| | | | |    |  __| `--. \`--. \ | | | . ` | | __ 
# | |   | |\ \| |___| |   | |\ \\ \_/ / \__/\| |___/\__/ /\__/ /_| |_| |\  | |_\ \
# \_|   \_| \_\____/\_|   \_| \_|\___/ \____/\____/\____/\____/ \___/\_| \_/\____/
                                                                                
                                                                                
# ______  ___ _____ ___    _____ _   _  ___   _     _____ _______   __            
# |  _  \/ _ \_   _/ _ \  |  _  | | | |/ _ \ | |   |_   _|_   _\ \ / /            
# | | | / /_\ \| |/ /_\ \ | | | | | | / /_\ \| |     | |   | |  \ V /             
# | | | |  _  || ||  _  | | | | | | | |  _  || |     | |   | |   \ /              
# | |/ /| | | || || | | | \ \/' / |_| | | | || |_____| |_  | |   | |              
# |___/ \_| |_/\_/\_| |_/  \_/\_\\___/\_| |_/\_____/\___/  \_/   \_/     
##############################################################################################################################


def excise_low_nsample_per_pol(uvd_x: UVData,
                               uvd_auto: UVData | None = None,
                               thresh: int = 1):
    """
    Remove only those (ant1, ant2, pol) keys whose max(nsamples) ≤ thresh.

    Returns
    -------
    new_x : UVData
    new_auto : UVData | None
    """
    # ------------------------------------------------------------------
    # 1.  good cross bls,pol
    # ------------------------------------------------------------------
    keep_keys = []                       # list of (a,b,pol) that survive
    for ant1, ant2, pol in uvd_x.get_antpairpols():
        ns = uvd_x.get_nsamples((ant1, ant2, pol))
        if np.nanmax(ns) > thresh:
            keep_keys.append((ant1, ant2, pol))
    print("keep_keys ",keep_keys)

    # ------------------------------------------------------------------
    # 2.  cross data selection with keep_keys
    # ------------------------------------------------------------------
    new_x = deepcopy(uvd_x)
    if keep_keys:                        # avoid empty selection
        new_x.select(
            bls=keep_keys,               # length-3 tuples are OK here
            keep_all_metadata=False,
            inplace=True,
        )
    else:                                # nothing survives ⇒ empty UVData
        raise RuntimeError("All cross keys were rejected by nsample threshold.")

    # ------------------------------------------------------------------
    # 3.  select autos, dropping only (a,a,pol) if that pol disappeared
    # ------------------------------------------------------------------
    new_auto = None
    if uvd_auto is not None:
        # Build a set of surviving cross keys per antenna & pol
        survived = {}
        for a, b, pol in keep_keys:
            survived.setdefault(a, set()).add(pol)
            survived.setdefault(b, set()).add(pol)

        keep_auto_keys = [
            (a, a, pol)
            for a, a2, pol in uvd_auto.get_antpairpols()
            if pol in survived.get(a, set())
        ]
        print("keep_auto_keys ", keep_auto_keys)

        new_auto = deepcopy(uvd_auto)
        if keep_auto_keys:
            new_auto.select(
                bls=keep_auto_keys,
                keep_all_metadata=False,
                inplace=True,
            )
        else:
            raise RuntimeError("All autos dropped by nsample criterion.")

    return new_x, new_auto


if cut_switch == 1:
    print("Excising low nsample data : True ")
    uvd_cleaned_inp_nsmp_cut, uvd_cleaned_inp_nsmp_auto_cut = excise_low_nsample_per_pol(
        uvd_cleaned_inp_nsmp,
        uvd_cleaned_inp_nsmp_auto,
        thresh=nsample_excise_thresh
    )
else:
    print("Excising low nsample data : False ")
    uvd_cleaned_inp_nsmp_cut = deepcopy(uvd_cleaned_inp_nsmp)
    uvd_cleaned_inp_nsmp_auto_cut = deepcopy(uvd_cleaned_inp_nsmp_auto)


redgrp_unpol_comb_cut = uvd_cleaned_inp_nsmp_cut.get_antpairs()
autos_set_cut = uvd_cleaned_inp_nsmp_auto_cut.get_antpairs()

print("old cross baselines:", len(uvd_cleaned_inp_nsmp.get_antpairpols()) )
print("old auto baselines :", len(uvd_cleaned_inp_nsmp_auto.get_antpairpols()) )

print("remaining cross baselines:", len(uvd_cleaned_inp_nsmp_cut.get_antpairpols()))
print("remaining auto baselines :", len(uvd_cleaned_inp_nsmp_auto_cut.get_antpairpols()))

print("old bls x", len(redgrp_unpol_comb), redgrp_unpol_comb)
print("old bls auto", len(autos_set), autos_set)

print("new bls x", len(redgrp_unpol_comb_cut), redgrp_unpol_comb_cut)
print("new bls auto", len(autos_set_cut), autos_set_cut)


print("2 _________ ",uvd_cleaned_inp_nsmp_cut .get_antpairs() )
print("3 _________ ",uvd_cleaned_inp_nsmp_auto_cut.get_antpairs() )

print("5 _________ ",uvd_cleaned_inp_nsmp_cut .get_antpairpols() )
print("6 _________ ",uvd_cleaned_inp_nsmp_auto_cut.get_antpairpols() )


baselines = uvd_cleaned_inp_nsmp_cut .get_antpairs()
num_baselines = len(baselines)
print("num_baselines ", num_baselines)

# ===========================================================================
# 10. REDUNDANT VISIBILITY AVERAGE  (coherent averaging of baselines)
# ===========================================================================
#
# Baselines within one redundant group sample (ideally) the same sky mode, so we
# average their COMPLEX visibilities coherently to boost SNR before estimating
# the power spectrum:
#
#     V_red(f, t) = ( sum_i  w_i(f,t) V_i(f,t) ) / ( sum_i  w_i(f,t) )
#
# where the sum runs over the N baselines in the group and w_i are the per-sample
# weights. Coherent averaging preserves the (redundant) sky signal while reducing
# thermal noise as ~1/sqrt(N) -- but any non-redundancy leaks signal, which is
# exactly the loss this script is built to check.
####################################################################################################################################################################################################################################################################################
# Redundant Average across Visibilities 
# ______ ___________ _   _ _   _______  ___   _   _ _____ _   __   __
# | ___ \  ___|  _  \ | | | \ | |  _  \/ _ \ | \ | |_   _| |  \ \ / /
# | |_/ / |__ | | | | | | |  \| | | | / /_\ \|  \| | | | | |   \ V / 
# |    /|  __|| | | | | | | . ` | | | |  _  || . ` | | | | |    \ /  
# | |\ \| |___| |/ /| |_| | |\  | |/ /| | | || |\  | | | | |____| |  
# \_| \_\____/|___/  \___/\_| \_/___/ \_| |_/\_| \_/ \_/ \_____/\_/  
                                                                   
                                                                   
#   ___  _   _ ___________  ___  _____  _____   _   _ _   _______    
#  / _ \| | | |  ___| ___ \/ _ \|  __ \|  ___| | | | | | | |  _  \   
# / /_\ \ | | | |__ | |_/ / /_\ \ |  \/| |__   | | | | | | | | | |   
# |  _  | | | |  __||    /|  _  | | __ |  __|  | | | | | | | | | |   
# | | | \ \_/ / |___| |\ \| | | | |_\ \| |___  | |_| \ \_/ / |/ /    
# \_| |_/\___/\____/\_| \_\_| |_/\____/\____/   \___/ \___/|___/     
#                                                                 
##############################################################################################################################
    

from hera_cal import utils

# red_avg_uvd = utils.red_average(uvd_cleaned_inp, reds=[redgrp_unpol_comb], propagate_flags=True)
red_avg_uvd = utils.red_average(uvd_cleaned_inp_nsmp_cut, propagate_flags=True)

summary = summarise_uvd_quality(red_avg_uvd)
print(summary)

# Quick RAM check after the (memory-heavy) redundant average.
show_ram()

# ===========================================================================
# 11. COHERENT POWER SPECTRUM  (uvp)
# ===========================================================================
#
# Delay-transform power spectrum of the redundantly-averaged visibilities. With a
# Blackman-Harris taper phi(f) and identity weighting, the per-baseline estimator
# (schematically) is
#
#     P(k_par) = X^2 Y * | FT_f[ phi(f) V(f) ] |^2 / (normalization)
#
# where FT_f is the frequency->delay Fourier transform and X^2 Y carries the
# cosmological (delay,freq)->(k_par) and beam/bandwidth scaling supplied via the
# beam object. The result `uvp` is the COHERENT (lossy) power-spectrum estimate.
# Instantiate a Cosmo Conversions object (default Planck cosmology); used to put
# the delay spectrum into cosmological units [mK^2 (Mpc/h)^3].
cosmo = hp.conversions.Cosmo_Conversions()
print(cosmo)
print("0")

print("spw_ranges ", spw_ranges)
print("verify flag array dimensions ", red_avg_uvd.flag_array.shape)

print("1")
print("3")

# Temporary on-disk pspec file (deleted later in the NaN-on-stats section).
output_file = f"Junk_{batchnum}_sim.h5"
output_name = os.path.join(save_path_H4C_SIM_NOISE_Chunked_RedGrp_NO_RedBlAvg_TAVG_FRF_PSPECH5_pI, output_file)

print("4")

# Read the primary beam (set in PARAMETERS) and wrap it for hera_pspec.
if not Path(beamfile).exists():
    raise FileNotFoundError(f"Beam file not found: {beamfile}")
beam = UVBeam()
beam.read_beamfits(beamfile)
print("Beam polarizations:", beam.polarization_array)
print("Beam data shape:", beam.data_array.shape)
print("data object pols reference ", red_avg_uvd.polarization_array)

uvb = hp.pspecbeam.PSpecBeamUV(beam, cosmo=cosmo)

# Reference spectral-window layout from the full FRF pipeline (kept for reference;
# the active estimate uses `spw_ranges` from the SPW section above).
SPWS_FRF = [(0, 95), (95, 180), (180, 265), (265, 365),
            (365, 417), (417, 497), (497, 577), (577, 657)]

# --- Estimate the coherent power spectrum from the redundantly-averaged data ---
ds = pspecdata.pspec_run(
    dsets=[red_avg_uvd],
    filename=output_name,
    spw_ranges=spw_ranges,
    pol_pairs=[polpair_in],
    input_data_weight=PSPEC_INPUT_DATA_WEIGHT,
    norm=PSPEC_NORM,
    taper=PSPEC_TAPER,
    file_type='uvh5',
    verbose=True,
    beam=uvb,
    overwrite=True,
    xant_flag_thresh=PSPEC_XANT_FLAG_THRESH,
    interleave_times=False,
    broadcast_dset_flags=True         # Included to bypass flag index error, 3 found, 4 expected : This flag is necessary to flag out frequncy gaps in a given LST; helps avoid ringing from delay transform
)

ds.Jy_to_mK()

# Because the LST integrations are offset by more than ~15 seconds we will get a warning
# but this is okay b/c it is still **significantly** less than the beam-crossing time and we are using short
# baselines...

# here we phase all datasets in dsets to the zeroth dataset
ds.rephase_to_dset(0)

# change units of UVData objects
ds.dsets[0].vis_units = 'mK'

# Specify which baselines to include (single coherent-average baseline here).
baselines = red_avg_uvd.get_antpairs()
print("baselines ", baselines)

# __Note : this function is used to access UVP object in RAM instead of write to h5 file; this means the baselines x baselines array would need to be set explicitly to incorporate all combinations; but for our purposes there is just one baseline (coherent average), hence the 'hack' is legit__

# Produce the power spectra drawn from dsets[0] across the spectral window(s),
# with identity weighting and a blackman-harris taper (taper applies across the
# delay axis and introduces correlations).
uvp = ds.pspec(baselines, baselines, (0, 0),
               [polpair_in],
               spw_ranges=spw_ranges, input_data_weight=PSPEC_INPUT_DATA_WEIGHT,
               norm=PSPEC_NORM,
               taper=PSPEC_TAPER,
               verbose=True)

print((uvp.get_dlys(0)))
print(len(uvp.get_dlys(0)))

tarr = uvp.lst_avg_array
tarrq = np.unique(tarr)*12/np.pi

print(len(tarr), "tarr ", tarr)
print(len(tarrq), "tarrq ", tarrq)

print(uvp)

# --- Working-directory / scratch listing (notebook !pwd / !ls, via subprocess) ---
print(subprocess.run(["pwd"], capture_output=True, text=True).stdout)
print(subprocess.run(["ls", "/lustre/aoc/projects/hera/rchandra/"], capture_output=True, text=True).stdout)

# --- Inspect the coherent UVPSpec (delays, LSTs, in-RAM size) ---
from pympler import asizeof

print((uvp.get_dlys(0)))
print(len(uvp.get_dlys(0)))

tarr = uvp.lst_avg_array
tarrq = np.unique(tarr)*12/np.pi

print(len(tarr), "tarr ", tarr)
print(len(tarrq), "tarrq ", tarrq)

print(uvp)


deep_size = asizeof.asizeof(uvp)      # bytes
print("full size :", deep_size/1e6, "MB")

free_named_objects("ds")   # coherent PSpecData no longer needed

# ===========================================================================
# 12. COMBINE CROSS + AUTO BASELINES
# ===========================================================================
#
# The incoherent estimate needs the autos alongside the cross baselines so the
# autos can supply the thermal-noise covariance model. Concatenate them here.
uvd_cleaned_inp_all = uvd_cleaned_inp_nsmp_cut + uvd_cleaned_inp_nsmp_auto_cut

deep_size = asizeof.asizeof(uvd_cleaned_inp_all)      # bytes
print("full size :", deep_size/1e6, "MB")

# Quick RAM check after building the combined object.
show_ram()

# ===========================================================================
# 13. INCOHERENT POWER SPECTRUM  (uvpinc)
# ===========================================================================
#
# Power spectrum formed per individual baseline (cross-baseline PAIRS are
# excluded; exclude_cross_bls=True), with the noise power computed from the autos
# (cov_model="autos"). The autos give the system temperature, from which the
# noise power spectrum P_N follows (schematically)
#
#     P_N  ~  Tsys^2 / ( sqrt(N_incoherent) * t_int )
#
# This is the reference, ~loss-free estimate to compare against the coherent uvp.

####################################################################################################################################################################################################################################################################################
# Incoherent Average 
#  _____ _   _ _____ _____ _   _  ___________ _____ _   _ _____  
# |_   _| \ | /  __ \  _  | | | ||  ___| ___ \  ___| \ | |_   _| 
#   | | |  \| | /  \/ | | | |_| || |__ | |_/ / |__ |  \| | | |   
#   | | | . ` | |   | | | |  _  ||  __||    /|  __|| . ` | | |   
#  _| |_| |\  | \__/\ \_/ / | | || |___| |\ \| |___| |\  | | |   
#  \___/\_| \_/\____/\___/\_| |_/\____/\_| \_\____/\_| \_/ \_/   
                                                               
                                                               
# ______  ___________ _____ _____                                
# | ___ \/  ___| ___ \  ___/  __ \                               
# | |_/ /\ `--.| |_/ / |__ | /  \/                               
# |  __/  `--. \  __/|  __|| |                                   
# | |    /\__/ / |   | |___| \__/\                               
# \_|    \____/\_|   \____/ \____/     
# 
##############################################################################################################################

print("verify flag array dimensions ", uvd_cleaned_inp_nsmp.flag_array.shape)

cosmo = hp.conversions.Cosmo_Conversions()
print(cosmo)
print("0")
print("1")
print("3")
print(uvb)

# Temporary on-disk pspec file (deleted later).
output_file = f"Junk_{batchnum}.h5"
output_name = os.path.join(save_path_H4C_SIM_NOISE_Chunked_RedGrp_NO_RedBlAvg_TAVG_FRF_PSPECH5_pI, output_file)

print("4")

# Beam objects `beam` / `uvb` were already read in the coherent section and reused here.
print("Beam polarizations:", beam.polarization_array)
print("Beam data shape:", beam.data_array.shape)
print("data object pols reference ", uvd_cleaned_inp.polarization_array)

dsinc = pspecdata.pspec_run(
    dsets=[uvd_cleaned_inp_all],
    filename=output_name,
    spw_ranges=spw_ranges,
    pol_pairs=[polpair_in],
    input_data_weight=PSPEC_INPUT_DATA_WEIGHT,
    norm=PSPEC_NORM,
    taper=PSPEC_TAPER,
    file_type='uvh5',
    verbose=True,
    beam=uvb,
    overwrite=True,
    xant_flag_thresh=PSPEC_XANT_FLAG_THRESH,
    interleave_times=False,
    broadcast_dset_flags=True,         # Included to bypass flag index error, 3 found, 4 expected
    exclude_cross_bls=True,
    store_cov = STORE_INCOHERENT_COVARIANCE,
    store_cov_diag = STORE_INCOHERENT_COVARIANCE_DIAG,
    cov_model = INCOHERENT_COV_MODEL,
)

dsinc.Jy_to_mK()

# here we phase all datasets in dsets to the zeroth dataset
dsinc.rephase_to_dset(0)

# change units of UVData objects
dsinc.dsets[0].vis_units = 'mK'

print("2 _________ ", len(uvd_cleaned_inp_nsmp_cut .get_antpairs()), uvd_cleaned_inp_nsmp_cut .get_antpairs() )
print("3 _________ ",uvd_cleaned_inp_nsmp_auto_cut.get_antpairs() )
print("5 _________ ",uvd_cleaned_inp_nsmp_cut .get_antpairpols() )
print("6 _________ ",uvd_cleaned_inp_nsmp_auto_cut.get_antpairpols() )

baselines = uvd_cleaned_inp_nsmp_cut.get_antpairs()

# Produce per-baseline power spectra across the spectral window(s), identity
# weighting, blackman-harris taper, storing the autos-based covariance.
uvpinc = dsinc.pspec(baselines, baselines, (0, 0),
                     [polpair_in],
                     spw_ranges=spw_ranges, input_data_weight=PSPEC_INPUT_DATA_WEIGHT,
                     norm=PSPEC_NORM,
                     taper=PSPEC_TAPER,
                     verbose=True,
                     store_cov = STORE_INCOHERENT_COVARIANCE,
                     store_cov_diag = STORE_INCOHERENT_COVARIANCE_DIAG,
                     cov_model = INCOHERENT_COV_MODEL,
                    )

# Output results
print("Averaged UVPSpec object:", uvpinc)

# View uvpinc contents behaviour
##############################################################################################################################
summarise_nan_stats(uvpinc, max_stats=2, max_spws=3, max_indices=4)

frac_dict = stats_nan_fraction_per_blpair(uvpinc)
tot_num=0
NaN_num=0
noNaN_num=0
for blp, pct in frac_dict.items():
    blpair_notation = uvp.blpair_to_antnums(blp)
    print(f"Baseline-pair {blpair_notation}: {pct:.2f}% NaNs")
    tot_num=tot_num+1
    if pct==0:
        noNaN_num=noNaN_num+1
    if pct>0:
        NaN_num=NaN_num+1
print("tot_num ", tot_num)
print("ratio no NaN", f"{noNaN_num/tot_num*100:.2f}" )

frac_dict = data_nan_fraction_per_blpair(uvpinc)
tot_num=0
NaN_num=0
noNaN_num=0
for blp, pct in frac_dict.items():
    print(f"Baseline-pair {blp}: {pct:.2f}% NaNs in data_array")
    tot_num=tot_num+1
    if pct==0:
        noNaN_num=noNaN_num+1
    if pct>0:
        NaN_num=NaN_num+1
print("tot_num ", tot_num)
print("ratio no NaN", f"{noNaN_num/tot_num*100:.2f}" )

PN = uvpinc.stats_array['autos_diag']

# print(PN['0'].shape) 

for spw, pn_array in uvpinc.stats_array['autos_diag'].items():
    print(f"Spectral window {spw} - PN shape: {pn_array.shape}")
##############################################################################################################################

free_named_objects("dsinc")   # incoherent PSpecData no longer needed

# ===========================================================================
# 14. NaN -> 0 ON STATS / NOISE ARRAYS (UVPSpec)
# ===========================================================================
#
# Zero out any NaNs in the UVPSpec arrays (data, integrations, weights, nsamples,
# stats) so the subsequent incoherent average is well defined, then delete the
# temporary "Junk" pspec file written by pspec_run().
####################################################################################################################################################################################################################################################################################
# NaN to 0 for stats_array/PNs :
# ______ _   _   _   _       _   _                   
# | ___ \ \ | | | \ | |     | \ | |                  
# | |_/ /  \| | |  \| | __ _|  \| |                  
# |  __/| . ` | | . ` |/ _` | . ` |                  
# | |   | |\  | | |\  | (_| | |\  |                  
# \_|   \_| \_/ \_| \_/\__,_\_| \_/                  
                                                   
                                                   
#  _   _   ___   _   _______ _     _____ _   _ _____ 
# | | | | / _ \ | \ | |  _  \ |   |_   _| \ | |  __ \
# | |_| |/ /_\ \|  \| | | | | |     | | |  \| | |  \/
# |  _  ||  _  || . ` | | | | |     | | | . ` | | __ 
# | | | || | | || |\  | |/ /| |_____| |_| |\  | |_\ \
# \_| |_/\_| |_/\_| \_/___/ \_____/\___/\_| \_/\____/
# 
##############################################################################################################################

def compute_nan_fraction_uvp(uvp):
    """
    Compute total NaN counts and fraction for each main array in a UVPSpec.
    Returns a dict: { array_name: (nan_count, total_count, fraction) }
    """
    summary = {}
    for attr in ['data_array', 'integration_array', 'wgt_array', 'nsample_array']:
        if hasattr(uvp, attr):
            arr_dict = getattr(uvp, attr)
            total = sum(arr.size for arr in arr_dict.values())
            nans = sum(np.isnan(arr).sum() for arr in arr_dict.values())
            fraction = nans / total if total else 0.0
            summary[attr] = (nans, total, fraction)
    if hasattr(uvp, 'stats_array'):
        total = 0
        nans = 0
        for spw_dict in uvp.stats_array.values():
            for arr in spw_dict.values():
                total += arr.size
                nans += np.isnan(arr).sum()
        fraction = nans / total if total else 0.0
        summary['stats_array'] = (nans, total, fraction)
    return summary

summary_before = compute_nan_fraction_uvp(uvpinc)
for name in summary_before:
    b_nans, b_total, b_frac = summary_before[name]
    print(f"{name}: before {b_nans}/{b_total} ({b_frac:.2%}), "
#           f"after {a_nans}/{a_total} ({a_frac:.2%})"
         )
    
def replace_nan_with_zero_uvp(uvp, inplace=False):
    """
    Return a copy of a UVPSpec with all NaNs zeroed,
    plus before/after NaN summaries for main arrays.
    """
    # Summary before cleaning
    summary_before = compute_nan_fraction_uvp(uvp)

    # Clean (in place if requested, else on a deep copy)
    uvp_clean = uvp if inplace else deepcopy(uvp)
    def clean_dict(arr_dict):
        for key, arr in arr_dict.items():
            arr_dict[key] = np.nan_to_num(arr, nan=0.0)
    for attr in ['data_array', 'integration_array', 'wgt_array', 'nsample_array']:
        if hasattr(uvp_clean, attr):
            clean_dict(getattr(uvp_clean, attr))
    if hasattr(uvp_clean, 'stats_array'):
        for stat, spw_dict in uvp_clean.stats_array.items():
            clean_dict(spw_dict)
    
    # Summary after cleaning
    summary_after = compute_nan_fraction_uvp(uvp_clean)

    return uvp_clean, summary_before, summary_after

uvp_clean, before, after = replace_nan_with_zero_uvp(uvpinc, inplace=CLEAN_UVPINC_INPLACE)
for name in before:
    b_nans, b_total, b_frac = before[name]
    a_nans, a_total, a_frac = after[name]
    print(f"{name}: before {b_nans}/{b_total} ({b_frac:.2%}), "
          f"after {a_nans}/{a_total} ({a_frac:.2%})")
    
uvpinc = uvp_clean if CLEAN_UVPINC_INPLACE else deepcopy(uvp_clean)

summary_before = compute_nan_fraction_uvp(uvpinc)
for name in summary_before:
    b_nans, b_total, b_frac = summary_before[name]
    print(f"{name}: before {b_nans}/{b_total} ({b_frac:.2%}), "
#           f"after {a_nans}/{a_total} ({a_frac:.2%})"
         )
    
    
import psutil, os
rss = psutil.Process(os.getpid()).memory_info().rss    # bytes in RAM
print(f"Resident memory: {rss/1e6:.2f} MB")
    

##############################################################################################################################


PN = uvpinc.stats_array['autos_diag']

# print(PN['0'].shape) 

for spw, pn_array in uvpinc.stats_array['autos_diag'].items():
    print(f"Spectral window {spw} - PN shape: {pn_array.shape}")

PN = uvpinc.stats_array['autos_diag']

# print(PN['0'].shape) 

for spw, pn_array in uvpinc.stats_array['autos_diag'].items():
    print(f"Spectral window {spw} - PN shape: {pn_array.shape}")
    if(spw==7):
        print(pn_array[0:10,45,:])
        print(pn_array[10:20,45,:])
        
        
####################################################################################################################################################################################################################################################################################
# Delete junk file created by pspec_run()

# check it exists first (optional)
if os.path.isfile(output_name):
    try:
        os.remove(output_name)
        print(f"Deleted {output_name}")
    except OSError as e:
        print(f"Error deleting {output_name}: {e}")
else:
    print(f"File not found: {output_name}")

# ===========================================================================
# 15. INCOHERENT AVERAGE OF POWER SPECTRA  (uvpspec_averaged)
# ===========================================================================
#
# Average the per-baseline power spectra together (an "incoherent" average of
# POWER, not visibilities). The weighting depends on `error_weights`:
#   - error_weights set      -> inverse-variance weighting from stats_array:
#         P_avg = ( sum_i  P_i / sigma_i^2 ) / ( sum_i  1 / sigma_i^2 )
#   - error_weights = None   -> integration-time / Nsample weighting (the ACTIVE
#     case here, since the call below only sets error_field, not error_weights):
#         w_i = ( t_int,i * sqrt(Nsamp_i) )^2 ,   P_avg = ( sum_i w_i P_i ) / ( sum_i w_i )
# `error_field='autos_diag'` only tells the routine which stats field to propagate
# (shrink) through the average; it does NOT switch on inverse-variance weighting.
#
# `average_spectra` below is the hera_pspec.grouping routine reproduced locally
# with extra diagnostic prints (logic unchanged); it is used here because of the
# in-RAM, all-baselines-in-one-group "hack" the notebook relies on.
####################################################################################################################################################################################################################################################################################
# Take Incoherent Average of PSPECs

from hera_pspec import utils as pspec_utils
from collections import OrderedDict as odict
from hera_pspec import uvpspec_utils as uvputils

def average_spectra(uvp_in, blpair_groups=None, time_avg=False,
                    blpair_weights=None, error_field=None,
                    error_weights=None, normalize_weights=True,
                    inplace=True, add_to_history=''):
    """
    Average power spectra across the baseline-pair-time axis, weighted by
    each spectrum's integration time or a specified kind of error bars.

    This is an "incoherent" average, in the sense that this averages power
    spectra, rather than visibility data. The 'nsample_array' and
    'integration_array' will be updated to reflect the averaging.

    In the case of averaging across baseline pairs, the resultant averaged
    spectrum is assigned to the zeroth blpair in the group. In the case of
    time averaging, the time and LST arrays are assigned to the mean of the
    averaging window.

    Note that this is designed to be separate from spherical binning in k:
    here we are not connecting k_perp modes to k_para modes. However, if
    blpairs holds groups of iso baseline separation, then this is
    equivalent to cylindrical binning in 3D k-space.

    If you want help constructing baseline-pair groups from baseline
    groups, see self.get_blpair_groups_from_bl_groups.

    Parameters
    ----------
    uvp_in : UVPSpec
        Input power spectrum (to average over).

    blpair_groups : list of baseline-pair groups
        List of list of tuples or integers. All power spectra in a
        baseline-pair group are averaged together. If a baseline-pair
        exists in more than one group, a warning is raised.

        Ex: blpair_groups = [ [((1, 2), (1, 2)), ((2, 3), (2, 3))],
                              [((4, 6), (4, 6))]]
        or blpair_groups = [ [1002001002, 2003002003], [4006004006] ]

    time_avg : bool, optional
        If True, average power spectra across the time axis. Default: False.

    blpair_weights : list of weights (float or int), optional
        Relative weight of each baseline-pair when performing the average. This
        is useful for bootstrapping. This should have the same shape as
        blpair_groups if specified. The weights are automatically normalized
        within each baseline-pair group. Default: None (all baseline pairs have
        unity weights).

    error_field: string or list, optional
        If errorbars have been entered into stats_array, will do a weighted
        sum to shrink the error bars down to the size of the averaged
        data_array. Error_field strings be keys of stats_array. If list,
        does this for every specified key. Every stats_array key that is
        not specified is thrown out of the new averaged object.

    error_weights: string, optional
         error_weights specify which kind of errors we use for weights
         during averaging power spectra.
         The weights are defined as $w_i = 1/ sigma_i^2$,
         where $sigma_i$ is taken from the relevant field of stats_array.
         If `error_weight' is set to None, which means we just use the
         integration time as weights. If error_weights is specified,
         then it also gets appended to error_field as a list.
         Default: None

    normalize_weights: bool, optional
        Whether to normalize the baseline-pair weights so that:
           Sum(blpair_weights) = N_blpairs
        If False, no normalization is applied to the weights. Default: True.

    inplace : bool, optional
        If True, edit data in self, else make a copy and return. Default:
        True.

    add_to_history : str, optional
        Added text to add to file history.

    Notes
    -----
    Currently, every baseline-pair in a blpair group must have the same
    Ntimes, unless time_avg=True. Future versions may support
    baseline-pair averaging of heterogeneous time arrays. This includes
    the scenario of repeated blpairs (e.g. in bootstrapping), which will
    return multiple copies of their time_array.
    """
    if inplace:
        uvp = uvp_in
    else:
        uvp = copy.deepcopy(uvp_in)

    # Copy these, so we don't modify the input lists
    blpair_groups = copy.deepcopy(blpair_groups)
    blpair_weights = copy.deepcopy(blpair_weights)

    # If blpair_groups were fed in, enforce type and structure
    if blpair_groups is not None:

        # Enforce shape of blpair_groups
        assert isinstance(blpair_groups[0], (list, np.ndarray)), \
              "blpair_groups must be fed as a list of baseline-pair lists. " \
              "See docstring."

        # Convert blpair_groups to list of blpair group integers
        if isinstance(blpair_groups[0][0], tuple):
            new_blpair_grps = [[uvp.antnums_to_blpair(blp) for blp in blpg]
                               for blpg in blpair_groups]
            blpair_groups = new_blpair_grps

        # Get all baseline pairs in uvp object (in integer form)
        uvp_blpairs = [uvp.antnums_to_blpair(blp) for blp in uvp.get_blpairs()]
        blvecs_groups = []
        for group in blpair_groups:
            blvecs_groups.append(uvp.get_blpair_blvecs()[uvp_blpairs.index(group[0])])
        # get baseline length for each group of baseline pairs
        # assuming only redundant baselines are paired together
        blpair_lens, _ = pspec_utils.get_bl_lens_angs(blvecs_groups, bl_error_tol=1.)

    else:
        # If not, each baseline pair is its own group
        _, idx = np.unique(uvp.blpair_array, return_index=True)
        blpair_groups = [[blp] for blp in uvp.blpair_array[np.sort(idx)]]
        # get baseline length for each group of baseline pairs
        # assuming only redundant baselines are paired together
        blpair_lens = [blv for blv in uvp.get_blpair_seps()[np.sort(idx)]]
        assert blpair_weights is None, "Cannot specify blpair_weights if "\
                                       "blpair_groups is None."

    # Print warning if a blpair appears more than once in all of blpair_groups
    all_blpairs = [item for sublist in blpair_groups for item in sublist]
    if len(set(all_blpairs)) < len(all_blpairs):
        print("Warning: some baseline-pairs are repeated between blpair "\
              "averaging groups.")

    # Create baseline-pair weights list if not specified
    if blpair_weights is None:
        # Assign unity weights to baseline-pair groups that were specified
        blpair_weights = [[1. for item in grp] for grp in blpair_groups]
    else:
        # Check that blpair_weights has the same shape as blpair_groups
        for i, grp in enumerate(blpair_groups):
            try:
                len(blpair_weights[i]) == len(grp)
            except:
                raise IndexError("blpair_weights must have the same shape as "
                                 "blpair_groups")

    # pre-check for error_weights
    if error_weights is None:
        use_error_weights = False
    else:
        if hasattr(uvp, "stats_array"):
            if error_weights not in uvp.stats_array.keys():
                raise KeyError("error_field \"%s\" not found in stats_array keys." % error_weights)
        use_error_weights = True

    # stat_l is a list of supplied error_fields, to sum over.
    if isinstance(error_field, (list, tuple, np.ndarray)):
        stat_l = list(error_field)
    elif isinstance(error_field, str):
        stat_l = [error_field]
    else:
        stat_l = []
    if use_error_weights:
        if error_weights not in stat_l:
            stat_l.append(error_weights)
    for stat in stat_l:
        if hasattr(uvp, "stats_array"):
            if stat not in uvp.stats_array.keys():
                raise KeyError("error_field \"%s\" not found in stats_array keys." % stat)

    if not uvp.exact_windows:
        # For baseline pairs not in blpair_groups, add them as their own group
        extra_blpairs = set(uvp.blpair_array) - set(all_blpairs)
        blpair_groups += [[blp] for blp in extra_blpairs]
        blpair_weights += [[1.,] for blp in extra_blpairs]

    # Create new data arrays
    data_array, wgts_array = odict(), odict()
    ints_array, nsmp_array = odict(), odict()
    stats_array = odict([[stat, odict()] for stat in stat_l])
    # will average covariance array if present
    store_cov = hasattr(uvp, "cov_array_real")
    if store_cov:
        cov_array_real = odict()
        cov_array_imag = odict()

    # same for window function
    store_window = hasattr(uvp, 'window_function_array')
    if store_window:
        window_function_array = odict()
        window_function_kperp, window_function_kpara = odict(), odict()
        
    w_list_all = []
    # Iterate over spectral windows
    for spw in range(uvp.Nspws):
        spw_data, spw_wgts, spw_ints, spw_nsmp = [], [], [], []
        spw_stats = odict([[stat, []] for stat in stat_l])
        if store_window:
            spw_window_function = []
            spw_wf_kperp_bins, spw_wf_kpara_bins = [], []
        if store_cov:
            spw_cov_real = []
            spw_cov_imag = []
            
        w_list_pol = []
        # Iterate over polarizations
        for i, p in enumerate(uvp.polpair_array):
            pol_data, pol_wgts, pol_ints, pol_nsmp = [], [], [], []
            pol_stats = odict([[stat, []] for stat in stat_l])
            if store_window:
                pol_window_function = []
            if store_cov:
                pol_cov_real = []
                pol_cov_imag = []

            # Iterate over baseline-pair groups
            for j, blpg in enumerate(blpair_groups):
                bpg_data, bpg_wgts, bpg_ints, bpg_nsmp = [], [], [], []
                bpg_stats = odict([[stat, []] for stat in stat_l])
                if store_window:
                    bpg_window_function = []
                if store_cov:
                    bpg_cov_real = []
                    bpg_cov_imag = []
                w_list = []

                # Sum over all weights within this baseline group to get
                # normalization (if weights specified). The normalization is
                # calculated so that Sum (blpair wgts) = no. baselines.
                if blpair_weights is not None:
                    blpg_wgts = np.array(blpair_weights[j])
                    norm = np.sum(blpg_wgts) if normalize_weights else 1.

                    if norm <= 0.:
                        raise ValueError("Sum of baseline-pair weights in "
                                         "group %d is <= 0." % j)
                    blpg_wgts = blpg_wgts * float(blpg_wgts.size) / norm # Apply normalization
                else:
                    blpg_wgts = np.ones(len(blpg))

                # Iterate within a baseline-pair group and get weighted data
                for k, blp in enumerate(blpg):
                    # Get no. samples and construct integration weight
                    nsmp = uvp.get_nsamples((spw, blp, p))[:, None]
                    # shape of nsmp: (Ntimes, 1)
                    # print("nsmp ", nsmp)
                    data = uvp.get_data((spw, blp, p))
                    # shape of data: (Ntimes, Ndlys)
                    wgts = uvp.get_wgts((spw, blp, p))
                    # shape of wgts: (Ntimes, Nfreqs, 2)
                    # print("wgts ", wgts)
                    ints = uvp.get_integrations((spw, blp, p))[:, None]
                    # shape of ints: (Ntimes, 1)
                    # print("ints ", ints)
                    if store_window:
                        window_function = uvp.get_window_function((spw, blp, p))
                        # shape of window_function if approx.: (Ntimes, Ndlys, Ndlys)
                        # shape of window_function if exact: (Ntimes, Ndlys, Nkperp, Nkpara)
                    if store_cov:
                        cov_real = uvp.get_cov((spw, blp, p), component="real")
                        cov_imag = uvp.get_cov((spw, blp, p), component="imag")
                        # shape of cov: (Ntimes, Ndlys, Ndlys)
                    # Get squared statistic
                    errws = {}
                    for stat in stat_l:
                        errws[stat] = uvp.get_stats(stat, (spw, blp, p)).copy()
                        np.square(errws[stat], out=errws[stat], where=np.isfinite(errws[stat]))
                        # shape of errs: (Ntimes, Ndlys)

                    if use_error_weights:
                        # If use_error_weights==True, all arrays are weighted by a specified kind of errors,
                        # including the error_filed in stats_array and cov_array.
                        # For each power spectrum P_i with error_weights sigma_i,
                        # P_avg = \sum{ P_i / (sigma_i)^2 } / \sum{ 1 / (sigma_i)^2 }
                        # while for other variance or covariance terms epsilon_i stored in stats_array and cov_array,
                        # epsilon_avg = \sum{ (epsilon_i / (sigma_i)^4 } / ( \sum{ 1 / (sigma_i)^2 } )^2
                        # For reference: M. Tegmark 1997, The Astrophysical Journal Letters, 480, L87, Table 1, #3
                        # or J. Dillon 2014, Physical Review D, 89, 023002 , Equation 34.
                        stat_val = uvp.get_stats(error_weights, (spw, blp, p)).copy().real #shape (Ntimes, Ndlys)
                        np.square(stat_val, out=stat_val, where=np.isfinite(stat_val))
                        #corrects for potential nan values
                        stat_val = np.nan_to_num(stat_val, copy=False, nan=np.inf, posinf=np.inf)
                        w = np.real(1. / stat_val.clip(1e-40, np.inf))
                        # shape of w: (Ntimes, Ndlys)
                    else:
                        # Otherwise all arrays are averaged in a way weighted by the integration time,
                        # including the error_filed in stats_array and cov_array.
                        # Since P_N ~ Tsys^2 / sqrt{N_incoherent} t_int (see N. Kern, The Astrophysical Journal 888.2 (2020): 70, Equation 7),
                        # we choose w ~ P_N^{-2} ~ (ints * sqrt{nsmp})^2
                        # integ1 = dset1.integration_time[blts1] * nsamp1
                        w = ( (ints) * np.sqrt(nsmp))**2
                        # shape of w: (Ntimes, 1)
                    # print("incoh w ", w.shape, w)

                    # Take time average if desired
                    if time_avg:
                        wsum = np.sum(w, axis=0).clip(1e-40, np.inf)
                        data = (np.sum(data * w, axis=0) \
                                / wsum)[None]
                        wgts = (np.sum(wgts * w[:, :1, None], axis=0) \
                                / wsum[:1, None])[None]
                        # wgts has a shape of (Ntimes, Nfreqs, 2), while
                        # w has a shape of (Ntimes, Ndlys) or (Ntimes, 1)
                        # To handle with the case  when Nfreqs != Ntimes,
                        # we choose to multiply wgts with w[:,:1,None].
                        ints = (np.sum(ints * w, axis=0) \
                                / wsum)[None]
                        nsmp = np.sum(nsmp, axis=0)[None]
                        if store_window:
                            if uvp.exact_windows:
                                window_function = (np.sum(window_function * w[:, :, None, None], axis=0)\
                                                    / (wsum)[:, None, None])[None]
                            if not uvp.exact_windows:
                                window_function = (np.sum(window_function * w[:, :, None], axis=0) \
                                                   / (wsum)[:, None])[None]
                        if store_cov:
                            cov_real = (np.sum(cov_real * w[:, :, None] * w[:, None, :], axis=0) \
                                   / wsum[:, None] / wsum[None, :])[None]
                            cov_imag = (np.sum(cov_imag * w[:, :, None] * w[:, None, :], axis=0) \
                                   / wsum[:, None] / wsum[None, :])[None]
                        for stat in stat_l:
                            # clip errws to eliminate nan: inf * 0 yields nans
                            weighted_errws = errws[stat].clip(0, 1e40) * w**2
                            errws[stat] = (np.sum(weighted_errws, axis=0) \
                                           / wsum**2)[None]
                            # set near-zero errws to inf, as they should be
                            errws[stat][np.isclose(errws[stat], 0)] = np.inf
                        w = np.sum(w, axis=0)[None]
                        # Above we use the clip method for zero weights. A tolerance
                        # as low as 1e-40 works when using inverse square of noise power
                        # as weights.
                    # Add multiple copies of data for each baseline according
                    # to the weighting/multiplicity;
                    # while multiple copies are only added when bootstrap resampling
                    for m in range(int(blpg_wgts[k])):
                        bpg_data.append(data * w)
                        bpg_wgts.append(wgts * w[:, :1, None])
                        bpg_ints.append(ints * w)
                        bpg_nsmp.append(nsmp)
                        for stat in stat_l:
                            # clip errws for same reason above
                            bpg_stats[stat].append(errws[stat].clip(0, 1e40) * w**2)
                        if store_window:
                            if uvp.exact_windows:
                                bpg_window_function.append(window_function * w[:, :, None, None])
                            else:
                                bpg_window_function.append(window_function * w[:, :, None])
                        if store_cov:
                            bpg_cov_real.append(cov_real * w[:, :, None] * w[:, None, :])
                            bpg_cov_imag.append(cov_imag * w[:, :, None] * w[:, None, :])
                        w_list.append(w)
                print("spw, pol, w_list ", spw, i, "\n", np.array(w_list).shape, np.array(w_list)[:,0,:].squeeze())
         
                # normalize sum: clip to deal with w_list_sum == 0
                w_list_sum = np.sum(w_list, axis=0).clip(1e-40, np.inf)
                bpg_data = np.sum(bpg_data, axis=0) / w_list_sum
                bpg_wgts = np.sum(bpg_wgts, axis=0) / w_list_sum[:,:1, None]
                bpg_nsmp = np.sum(bpg_nsmp, axis=0)
                bpg_ints = np.sum(bpg_ints, axis=0) / w_list_sum
                if store_cov:
                    bpg_cov_real = np.sum(bpg_cov_real, axis=0) / w_list_sum[:, :, None] / w_list_sum[:, None, :]
                    bpg_cov_imag = np.sum(bpg_cov_imag, axis=0) / w_list_sum[:, :, None] / w_list_sum[:, None, :]
                for stat in stat_l:
                    stat_avg = np.sum(bpg_stats[stat], axis=0) / w_list_sum**2
                    # set near-zero stats to inf, as they should be
                    stat_avg[np.isclose(stat_avg, 0)] = np.inf
                    # take sqrt to get back to stat units
                    bpg_stats[stat] = np.sqrt(stat_avg)
                if store_window:
                    if uvp.exact_windows:
                        bpg_window_function = np.sum(bpg_window_function, axis=0) # / w_list_sum[:, :, None, None]
                    else:
                        bpg_window_function = np.sum(bpg_window_function, axis=0) / w_list_sum[:, :, None]
                # Append to lists (polarization)
                pol_data.extend(bpg_data); pol_wgts.extend(bpg_wgts)
                pol_ints.extend(bpg_ints); pol_nsmp.extend(bpg_nsmp)
                for stat in stat_l:
                    pol_stats[stat].extend(bpg_stats[stat])
                if store_window:
                    pol_window_function.extend(bpg_window_function)
                if store_cov:
                    pol_cov_real.extend(bpg_cov_real)
                    pol_cov_imag.extend(bpg_cov_imag)
                    
                w_list_pol.append(w_list)
            
            # Append to lists (spectral window)
            spw_data.append(pol_data); spw_wgts.append(pol_wgts)
            spw_ints.append(pol_ints); spw_nsmp.append(pol_nsmp)
            for stat in stat_l:
                spw_stats[stat].append(pol_stats[stat])
            if store_window:
                spw_window_function.append(pol_window_function)
                if uvp.exact_windows:
                    spw_wf_kperp_bins.append(uvp.window_function_kperp[spw][:, i])
                    spw_wf_kpara_bins.append(uvp.window_function_kpara[spw][:, i])
            if store_cov:
                spw_cov_real.append(pol_cov_real)
                spw_cov_imag.append(pol_cov_imag)
                
        w_list_all.append(w_list_pol)

        # Append to dictionaries
        data_array[spw] = np.moveaxis(spw_data, 0, -1)
        wgts_array[spw] = np.moveaxis(spw_wgts, 0, -1)
        ints_array[spw] = np.moveaxis(spw_ints, 0, -1)[:, 0, :]
        nsmp_array[spw] = np.moveaxis(spw_nsmp, 0, -1)[:, 0, :]
        for stat in stat_l:
            stats_array[stat][spw] = np.moveaxis(spw_stats[stat], 0, -1)
        if store_window:
            window_function_array[spw] = np.moveaxis(spw_window_function, 0, -1)
            if uvp.exact_windows:
                window_function_kperp[spw] = np.moveaxis(spw_wf_kperp_bins, 0, -1)
                window_function_kpara[spw] = np.moveaxis(spw_wf_kpara_bins, 0, -1)
        if store_cov:
            cov_array_real[spw] = np.moveaxis(np.array(spw_cov_real), 0, -1)
            cov_array_imag[spw] = np.moveaxis(np.array(spw_cov_imag), 0, -1)
            
    # print("w_list ", np.array(w_list).shape, np.array(w_list)[:,0,:].squeeze())
    # print("w_list_all ", np.array(w_list_all).shape, np.array(w_list_all)[:,0,:].squeeze())

    # Iterate over blpair groups one more time to assign metadata
    time_1, time_2, time_avg_arr  = [], [], []
    lst_1, lst_2, lst_avg_arr = [], [], []
    blpair_arr, bl_arr = [], []

    for i, blpg in enumerate(blpair_groups):

        # Get blpairts indices for zeroth blpair in this group
        blpairts = uvp.blpair_to_indices(blpg[0])

        # Assign meta-data
        bl_arr.extend(list(uvputils._blpair_to_bls(blpg[0])))
        if time_avg:
            blpair_arr.append(blpg[0])
            time_1.extend([np.mean(uvp.time_1_array[blpairts])])
            time_2.extend([np.mean(uvp.time_2_array[blpairts])])
            time_avg_arr.extend([np.mean(uvp.time_avg_array[blpairts])])
            lst_1.extend([np.mean(np.unwrap(uvp.lst_1_array[blpairts]))%(2*np.pi)])
            lst_2.extend([np.mean(np.unwrap(uvp.lst_2_array[blpairts]))%(2*np.pi)])
            lst_avg_arr.extend([np.mean(np.unwrap(uvp.lst_avg_array[blpairts]))%(2*np.pi)])
        else:
            blpair_arr.extend(np.ones_like(blpairts, int) * blpg[0])
            time_1.extend(uvp.time_1_array[blpairts])
            time_2.extend(uvp.time_2_array[blpairts])
            time_avg_arr.extend(uvp.time_avg_array[blpairts])
            lst_1.extend(uvp.lst_1_array[blpairts])
            lst_2.extend(uvp.lst_2_array[blpairts])
            lst_avg_arr.extend(uvp.lst_avg_array[blpairts])

    # Update arrays
    bl_arr = np.array(sorted(set(bl_arr)))
    bl_vecs = np.array([uvp.bl_vecs[uvp.bl_array.tolist().index(bl)]
                        for bl in bl_arr])

    # Assign arrays and metadata to UVPSpec object
    uvp.Ntimes = len(np.unique(np.hstack([time_1, time_2])))
    uvp.Nbltpairs = len(time_avg_arr)
    uvp.Nblpairs = len(np.unique(blpair_arr))
    uvp.Nbls = len(bl_arr)
    uvp.Ntpairs = len(set((t1, t2) for t1, t2 in zip(time_1, time_2)))

    # Baselines
    uvp.bl_array = bl_arr
    uvp.bl_vecs = bl_vecs
    uvp.blpair_array = np.array(blpair_arr)

    # Times
    uvp.time_1_array = np.array(time_1)
    uvp.time_2_array = np.array(time_2)
    uvp.time_avg_array = np.array(time_avg_arr)

    # LSTs
    uvp.lst_1_array = np.array(lst_1)
    uvp.lst_2_array = np.array(lst_2)
    uvp.lst_avg_array = np.array(lst_avg_arr)

    # Data, weights, and no. samples
    uvp.data_array = data_array
    uvp.integration_array = ints_array
    uvp.wgt_array = wgts_array
    # print("wgts_array ", np.array(wgts_array).shape, wgts_array)
    uvp.nsample_array = nsmp_array
    if store_window:
        uvp.window_function_array = window_function_array
        if uvp.exact_windows:
            uvp.window_function_kperp = window_function_kperp
            uvp.window_function_kpara = window_function_kpara
    if store_cov:
        uvp.cov_array_real = cov_array_real
        uvp.cov_array_imag = cov_array_imag
    if len(stat_l) >=1 :
        uvp.stats_array = stats_array
    elif hasattr(uvp, "stats_array"):
        delattr(uvp, "stats_array")

    # Add to history
    uvp.history = "Spectra averaged with hera_pspec [{}]\n{}\n{}\n{}".format(__version__, add_to_history, '-'*40, uvp.history)
    # Validity check
    uvp.check()

    # Return (always return uvp so inplace=True is safe; uvp IS uvp_in when inplace)
    return uvp
    
    

# from hera_pspec.grouping import average_spectra

# # Specify baseline-pair groups for averaging (adjust based on your data)
# blpair_groups = [[(ant1, ant2)]]  # List of lists grouping similar baseline pairs
# Get all unique baseline pairs from the UVPSpec object
all_blpairs = uvpinc.get_blpairs()

# Group all baseline pairs together in a single group for averaging
blpair_groups = [all_blpairs]  # A list with one sublist containing all baseline pairs
print("blpair_groups", len(all_blpairs), blpair_groups)
# blpair_groups = unique_to_each_ordered
# print("clean_baselines ", clean_baselines)

# Run average_spectra on the UVPSpec object
uvpspec_averaged = average_spectra(
    uvp_in=uvpinc,                  # UVPSpec object from pspec
    blpair_groups=blpair_groups, # List of baseline-pair groups
    error_field=INCOHERENT_AVERAGE_ERROR_FIELD,
    time_avg=INCOHERENT_AVERAGE_TIME,
    inplace=AVERAGE_SPECTRA_INPLACE,
)

# Output results
print("Averaged UVPSpec object:", uvpspec_averaged)


print(uvpspec_averaged.stats_array.keys() )

PN = uvpspec_averaged.stats_array['autos_diag']

# print(PN['0'].shape) 

for spw, pn_array in uvpspec_averaged.stats_array['autos_diag'].items():
    print(f"Spectral window {spw} - PN shape: {pn_array.shape}")
    if(spw==7):
        print(pn_array[0:10,45,:])
#         print(pn_array[10:20,45,:])

# ===========================================================================
# 16. FINAL NaN INSPECTION OF THE AVERAGED SPECTRUM
# ===========================================================================

summarise_nan_stats(uvpspec_averaged, max_stats=2, max_spws=3, max_indices=4)

frac_dict = stats_nan_fraction_per_blpair(uvpspec_averaged)
tot_num=0
NaN_num=0
noNaN_num=0
for blp, pct in frac_dict.items():
    print(f"Baseline-pair {blp}: {pct:.2f}% NaNs")
    tot_num=tot_num+1
    if pct==0:
        noNaN_num=noNaN_num+1
    if pct>0:
        NaN_num=NaN_num+1
print("tot_num ", tot_num)
print("ratio no NaN", f"{noNaN_num/tot_num*100:.2f}" )

frac_dict = data_nan_fraction_per_blpair(uvpspec_averaged)
tot_num=0
NaN_num=0
noNaN_num=0
for blp, pct in frac_dict.items():
    print(f"Baseline-pair {blp}: {pct:.2f}% NaNs in data_array")
    tot_num=tot_num+1
    if pct==0:
        noNaN_num=noNaN_num+1
    if pct>0:
        NaN_num=NaN_num+1
print("tot_num ", tot_num)
print("ratio no NaN", f"{noNaN_num/tot_num*100:.2f}" )

# ===========================================================================
# 17. SAVE OUTPUTS
# ===========================================================================
#
# Persist both products to PSpecContainer .h5 files in DATA_PATH:
#   uvp               -> coherent power spectrum
#   uvpspec_averaged  -> incoherent (blpair-averaged) power spectrum
# To save :
# Incoherently averaged PSPEC objects:
# uvpspec_averaged
# Coherently averaged PSPEC objects:
# uvp
# Other data products required for plotting 
# 1. LST 
# 2. spws, frequency channels
# 3. pol

pol=pol_in


save_path = f"output_vis_sigloss_check_out/Sig_Loss_{run_batch}_{pol}/"
# DATA_PATH
out_dir = os.path.dirname(save_path)
if out_dir:  # non‐empty (i.e. you did give a folder, not just a filename)
    os.makedirs(out_dir, exist_ok=True)


# Get just the last path component
base = MODEL_DIR.name
# Extract ideal tag: after "subset_" and before the next "_"
m_ideal = re.search(r"subset_([^_]+)_", base)
ideal_tag = m_ideal.group(1) if m_ideal else "ideal"
# Extract airy tag: after "nonred_" to the end (no more "_")
m_airy = re.search(r"nonred_([^_]+)$", base)
airy_tag = m_airy.group(1) if m_airy else "airy"
print("ideal_tag:", ideal_tag)  
print("airy_tag :", airy_tag)   
print("sky_type:", sky_type)
print("chunks ", f"_ck{chunk_min:05d}-{chunk_max:05d}")

print("DATA_PATH ", DATA_PATH)

output_file = f"notebook_{sky_type}_{ideal_tag}_{airy_tag}_Coh_PSPEC_{bl_len}_{bl_ang}_{pol}_fch{fch_min:04d}-{fch_max:04d}_ck{chunk_min:05d}-{chunk_max:05d}_{proc1[0]}_{proc2[0]}_bch_{batchnum}_psc_PN.h5"
print("output_file 1", output_file)
output_name = os.path.join(DATA_PATH, output_file)
psc = hp.PSpecContainer(output_name, mode='rw', keep_open=False)
psc.set_pspec('dset0', 'dset0_x_dset0', uvp, overwrite=True)

output_file = f"notebook_{sky_type}_{ideal_tag}_{airy_tag}_Incoh_PSPEC_{bl_len}_{bl_ang}_{pol}_fch{fch_min:04d}-{fch_max:04d}_ck{chunk_min:05d}-{chunk_max:05d}_{proc1[0]}_{proc2[0]}_bch_{batchnum}_psc_PN.h5"
print("output_file 2", output_file)
output_name = os.path.join(DATA_PATH, output_file)
psc = hp.PSpecContainer(output_name, mode='rw', keep_open=False)
psc.set_pspec('dset0', 'dset0_x_dset0', uvpspec_averaged, overwrite=True)


print("CODE HAS FINISHED")

print(save_path)
print(output_file)

# ===========================================================================
# 18. OPTIONAL: FILE-COMPLETENESS CHECK (independent diagnostic)
# ===========================================================================
#
# Reports any missing fch####_chunk#####.uvh5 files in `mfc_directory` over the
# (mfc_channels x mfc_chunks) grid. Uses its own parameters from the PARAMETERS
# section and does not affect the pipeline above.

check_missing_files(mfc_directory, mfc_channels, mfc_chunks)
