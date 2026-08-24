#!/usr/bin/env python
"""
output_vis_sigloss_check.py
============================
Curated script version of output_vis_sigloss_check.ipynb.

End-to-end signal-loss check pipeline:

    1. Read concatenated cross + auto UVH5 visibilities
    2. Clean NaNs, smooth nsamples, excise bad baselines
    3. Redundant-average visibilities
    4. Compute coherent power spectrum  (red-avg'd data × itself)
    5. Compute incoherent power spectrum (per-baseline, then average in P-space)
    6. Save both PSpec products to HDF5

Run as:
    python output_vis_sigloss_check.py

Or via SLURM (adjust params at the top).
"""

###############################################################################
#  ___  ___  ___  ____   ___  ____  _____  ____                               #
# |_ _||  \/  | |  _ \ / _ \|  _ \|_   _|/ ___|                              #
#  | | | |\/| | | |_) | | | | |_) | | |  \___ \                              #
#  | | | |  | | |  __/| |_| |  _ <  | |   ___) |                             #
# |___||_|  |_| |_|    \___/|_| \_\ |_|  |____/                              #
#                                                                             #
###############################################################################

import os
import gc
import re
import ctypes
import psutil
import itertools

import numpy as np
import matplotlib

from pathlib import Path
from copy import deepcopy
from collections import OrderedDict as odict

import hdf5plugin  # noqa: F401 - registers HDF5 compression filters when needed

from pyuvdata import UVData, UVBeam
import hera_pspec as hp
from hera_pspec import pspecdata
from hera_pspec import utils as pspec_utils
from hera_pspec import uvpspec_utils as uvputils
from hera_cal import utils

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = "18"


def malloc_trim():
    """Release glibc malloc arenas back to the OS (Linux only)."""
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def show_ram():
    rss = psutil.Process(os.getpid()).memory_info().rss
    print(f"RAM usage: {rss/1e9:.2f} GB")


def free_named_objects(*names):
    """Drop large top-level objects once downstream products are already made."""
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


# Print dependency versions
for repo in ['numpy', 'scipy', 'astropy', 'hera_cal', 'hera_qm',
             'hera_filters', 'pyuvdata', 'hera_pspec']:
    exec(f'from {repo} import __version__')
    print(f'{repo}: {__version__}')


###############################################################################
#   ____ _     ___  ____    _    _       ____   _    ____      _    __  __     #
#  / ___| |   / _ \| __ )  / \  | |     |  _ \ / \  |  _ \   / \  |  \/  |   #
# | |  _| |  | | | |  _ \ / _ \ | |     | |_) / _ \ | |_) | / _ \ | |\/| |   #
# | |_| | |__| |_| | |_) / ___ \| |___  |  __/ ___ \|  _ < / ___ \| |  | |   #
#  \____|_____\___/|____/_/   \_\_____| |_| /_/   \_\_| \_/_/   \_\_|  |_|   #
#                                                                             #
#  All tuneable knobs live here.  Change these before each run.               #
###############################################################################

# ── Run / batch identifiers ──────────────────────────────────────────────────
run_batch     = 250925          # date tag ddmmyy
total_cores   = 10
batch_number  = 5
batchnum      = 0               # used for scratch filename

# ── Baseline selection ───────────────────────────────────────────────────────
bl_len  = 14.0                  # target baseline length [m]
bl_ang  = 0.0                   # target angle [deg], e.g. EW group
len_tol = 2.0                   # ± length tolerance [m]
ang_tol = 5.0                   # ± angle tolerance [deg]
REDUNDANT_GROUP_BL_ERROR_TOL_META = 2.0
REDUNDANT_GROUP_BL_ERROR_TOL_DATA = 1.0
REDUNDANT_GROUP_LEN_RANGE = (10.0, 100.0)
REDUNDANT_GROUP_ANG_RANGE = (0.0, 180.0)

# ── Frequency channel range ──────────────────────────────────────────────────
fch_min, fch_max = 227, 315     # inclusive channel range
chunk_min, chunk_max = 0, 144   # inclusive chunk range
SPW_RANGES = None               # None => one SPW spanning all loaded channels

# ── Polarization ─────────────────────────────────────────────────────────────
pol_in = "xx"                   # one of {"xx", "yy", "xy", "yx"}

# ── Preprocessing switches ───────────────────────────────────────────────────
cut_switch = 1                  # 1 = excise baselines with zero nsamples

# ── Power-spectrum settings ──────────────────────────────────────────────────
# Coherent PSPEC: average redundant visibilities first, then delay transform.
#   P_coh(τ) = |F_ν{N^{-1} Σ_i V_i(ν)}|²
# Incoherent PSPEC: delay-transform each redundant visibility first, then avg.
#   P_inc(τ) = N^{-1} Σ_i |F_ν{V_i(ν)}|²
PSPEC_INPUT_DATA_WEIGHT = "identity"
PSPEC_NORM = "I"
PSPEC_TAPER = "blackman-harris"
PSPEC_XANT_FLAG_THRESH = 0.95
STORE_INCOHERENT_COVARIANCE = True
STORE_INCOHERENT_COVARIANCE_DIAG = True
INCOHERENT_COV_MODEL = "autos"
INCOHERENT_AVERAGE_ERROR_FIELD = "autos_diag"
INCOHERENT_AVERAGE_TIME = False

# ── Output-product labels ────────────────────────────────────────────────────
PROC_BASELINE_LABEL = "cutbl"
PROC_LST_LABEL = "cutlst"

# ── Memory/performance controls ──────────────────────────────────────────────
# The incoherent UVPSpec can contain very large data/covariance arrays.  These
# defaults avoid multi-GB deep copies once the object is no longer needed in its
# original, unaveraged form.
CLEAN_UVPINC_INPLACE = True
AVERAGE_SPECTRA_INPLACE = True
FREE_INTERMEDIATE_OBJECTS = True

# ── Sky type label ───────────────────────────────────────────────────────────
# sky_type = "ptsrc"
sky_type = "eor_ns"

# ── Base output directory ────────────────────────────────────────────────────
BASE_OUTDIR = Path("/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/outputs")
# BASE_OUTDIR = Path("/lustre/aoc/projects/hera/kmandar/repos/validation-sim/outputs")

# ── Model directory ──────────────────────────────────────────────────────────
# Uncomment ONE MODEL_DIR per run.
# The commented options below are retained as run documentation, because they
# encode the sky / beam / array combinations used across validation tests.
#
# ═══════════════════════════════════════════════════════════════════════════════
# PTSRC SKY
# ═══════════════════════════════════════════════════════════════════════════════
# Airy ─────────────────────────────────────────────────────────
# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# ideal ENU, diameter Airy var (deltaD ~ <.2m)        airyprb, idealT
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyprb")
# real ENU, diameter Airy var (deltaD ~ <.2m)         airyprb, idealF
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyprb")
# ideal ENU, tilt Airy var (deltZa ~ 2-3 degree)      airytilt, idealT
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")
# real ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealF
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")
# Vivaldi ──────────────────────────────────────────────────────
# ideal ENU, ideal Vivaldi       vivaldired, idealT
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
# real ENU, ideal Vivaldi       vivaldired, idealF
# MODEL_DIR = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
#
# ═══════════════════════════════════════════════════════════════════════════════
# EOR SKY
# ═══════════════════════════════════════════════════════════════════════════════
# Airy ─────────────────────────────────────────────────────────
# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# ideal ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealT
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")
# real ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealF
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")
# Vivaldi ──────────────────────────────────────────────────────
# ideal ENU, ideal Vivaldi       vivaldired, idealT
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
# real ENU, ideal Vivaldi       vivaldired, idealF
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
# Gaussian ─────────────────────────────────────────────────────
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
#
# ═══════════════════════════════════════════════════════════════════════════════
# EOR SKY PLAYGROUND
# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian ─────────────────────────────────────────────────────
# MODEL_DIR = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix_freqclone.03001951._beammapperant_airyred-nonred_airyred")
# MODEL_DIR = Path("eor-grf-256/rlzn_seed_111_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR = Path("eor-grf-256/rlzn_seed_111_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR = Path("eor-grf-256/rlzn_seed_111_offsetfix_spatmean_pwlw/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_80MHzref_gauss_nonspectral_skysd111_eoroffsetfix.29493475._beammapperant_airyred-nonred_airyred/")
MODEL_DIR  = Path("eor-grf-256/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")
# Airy ─────────────────────────────────────────────────────────
# MODEL_DIR = Path("eor-grf-256/rlzn_seed_111_offsetfix/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_skysd111_eoroffsetfix.5cfe1b35._beammapperant_airyred-nonred_airyred/")
# Isotropic ────────────────────────────────────────────────────
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_111_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_iso_skysd111_eoroffsetfix.920cf40b._beammapperant_airyred-nonred_airyred/")
#
# ═══════════════════════════════════════════════════════════════════════════════
# EOR Noisy
# ═══════════════════════════════════════════════════════════════════════════════
# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# MODEL_DIR = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR = Path("eor-grf-256-noisy-2x/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR = Path("eor-grf-256-noisy-correct-beam/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR = Path("eor-grf-256-noisy-D7m-airy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy.5cfe1b35._beammapperant_airyred-nonred_airyred/")      # seed 777
# MODEL_DIR = Path("eor-grf-256-noisy-D7m-airy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy_skysd111.5cfe1b35._beammapperant_airyred-nonred_airyred/")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
#
# ═══════════════════════════════════════════════════════════════════════════════
# PURE Noise
# ═══════════════════════════════════════════════════════════════════════════════
# MODEL_DIR = Path("noise-only-300k/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")

# ── Beam file ────────────────────────────────────────────────────────────────
beam_path = '/home/herastore02-1/HERA_Validation_rchandra/'
# Vivaldi
# beamfile = os.path.join(beam_path, 'NF_HERA_Vivaldi_efield_beam_healpix.fits')
# beamfile = os.path.join(beam_path, 'NF_HERA_Vivaldi_efield_beam_extrap_0.fits')

# Airy
# beamfile = os.path.join(beam_path, 'airy_beam_7.0m_decay_0.3dBdeg_start_70.0deg_healpix.fits')
# beamfile = os.path.join(beam_path, 'airy_beam_7.0m_freqconst_ref80MHz_decay_0.3dBdeg_start_70.0deg_healpix.fits')
beamfile = os.path.join(beam_path, 'airy_beam_14.0m_freqconst_ref80MHz_decay_0.3dBdeg_start_70.0deg_healpix.fits')

# Gaussian
# beamfile = os.path.join(beam_path, 'gaussian_beam_7.0m_decay_0.3dBdeg_start_70.0deg_healpix.fits')

# Vivaldi pI
# beamfile = os.path.join(beam_path, 'NF_HERA_Vivaldi_efield_beam_healpix_pstokes.fits')

# ── Scratch output path for temp pspec files ─────────────────────────────────
save_path_scratch = Path("/home/herastore02-1/H6C_scratch_rchandra/Sig_Loss_1_pI")

# ── Derived paths ────────────────────────────────────────────────────────────
DATA_PATH = BASE_OUTDIR / MODEL_DIR
save_path_scratch.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DATA_PATH:", DATA_PATH)
print("BEAM FILE:", beamfile)
print("SCRATCH PATH:", save_path_scratch)
print("=" * 70)


###############################################################################
#  ____  _____ ____    ____  ____   ___  _   _ ____                           #
# |  _ \| ____|  _ \  / ___|/ ___| / _ \| | | |  _ \                         #
# | |_) |  _| | | | || |  _| |  _ | |_) | | | | |_) |                        #
# |  _ <| |___| |_| || |_| | |_| ||  _ <| |_| |  __/                         #
# |_| \_\_____|____/  \____|\____||_| \_\\___/|_|                            #
#                                                                             #
#  Find the redundant baseline group nearest to (bl_len, bl_ang).             #
###############################################################################

def fname_for(ch, chunk):
    """Return full path to fch####_chunk#####.uvh5 for given channel, chunk."""
    return DATA_PATH / f"fch{ch:04d}_chunk{chunk:05d}.uvh5"


# Load metadata from the first file
ref_fn = fname_for(fch_min, chunk_min)
if not ref_fn.exists():
    raise FileNotFoundError(f"Reference file {ref_fn} not found.")

uvd_meta = UVData()
uvd_meta.read_uvh5(ref_fn, read_data=False)
print(f"Loaded metadata from: {ref_fn}")
print(f"Nants_data: {uvd_meta.Nants_data}   Nbls: {uvd_meta.Nbls}")

red_bls, lens, angs = hp.utils.get_reds(
    uvd_meta,
    bl_error_tol=REDUNDANT_GROUP_BL_ERROR_TOL_META,
    add_autos=True,
    bl_len_range=REDUNDANT_GROUP_LEN_RANGE,
    bl_deg_range=REDUNDANT_GROUP_ANG_RANGE,
    pick_data_ants=True,
)
print(f"Found {len(red_bls)} redundant groups.")

best_idx, best_score = None, None
for i, (L, A) in enumerate(zip(lens, angs)):
    if (abs(L - bl_len) <= len_tol) and (abs(A - bl_ang) <= ang_tol):
        score = (L - bl_len)**2 + (A - bl_ang)**2
        if best_score is None or score < best_score:
            best_score, best_idx = score, i

if best_idx is None:
    raise RuntimeError(
        f"No redundant group within {len_tol} m / {ang_tol} deg "
        f"of (L={bl_len}, A={bl_ang})."
    )

print(f"Selected group {best_idx}: "
      f"L = {lens[best_idx]:.2f} m, A = {angs[best_idx]:.2f}°")

redgrp = red_bls[best_idx]
redgrp_unpol = [(a1, a2) for a1, a2 in redgrp if a1 != a2]
autos_set = sorted({(a, a) for a1, a2 in redgrp for a in (a1, a2)})
print(f"Cross baselines: {len(redgrp_unpol)}   Autos: {len(autos_set)}")


###############################################################################
#  _     ___    _    ____    ____    _  _____  _                              #
# | |   / _ \  / \  |  _ \  |  _ \  / \|_   _|/ \                            #
# | |  | | | |/ _ \ | | | | | | | |/ _ \ | | / _ \                           #
# | |__| |_| / ___ \| |_| | | |_| / ___ \| |/ ___ \                          #
# |_____\___/_/   \_\____/  |____/_/   \_\_/_/   \_\                         #
#                                                                             #
#  Read the pre-concatenated cross + auto UVH5 files.                         #
###############################################################################

base = MODEL_DIR.name
path = MODEL_DIR.parent
print(f"MODEL_DIR base: {base}")
print(f"MODEL_DIR path: {path}")

m_ideal = re.search(r"subset_([^_]+)_", base)
ideal_tag = m_ideal.group(1) if m_ideal else "ideal"

m_airy = re.search(r"nonred_([^_]+)$", base)
airy_tag = m_airy.group(1) if m_airy else "airy"

print(f"ideal_tag: {ideal_tag}")
print(f"airy_tag : {airy_tag}")

tag_stem = (
    f"{ideal_tag}_{airy_tag}"
    f"_fch{fch_min:04d}-{fch_max:04d}"
    f"_ck{chunk_min:05d}-{chunk_max:05d}"
    f"_bl{bl_len:.1f}m"
)

cross_outfile = DATA_PATH / f"uvd_cross_{tag_stem}.uvh5"
autos_outfile = DATA_PATH / f"uvd_autos_{tag_stem}.uvh5"

print(f"Expected cross file: {cross_outfile}")
print(f"Expected autos file: {autos_outfile}")

if cross_outfile.exists() and autos_outfile.exists():
    print("\nReading concatenated files...")
    uvd_combined = UVData()
    uvd_combined.read_uvh5(cross_outfile)

    uvd_autos = UVData()
    uvd_autos.read_uvh5(autos_outfile)

    print(f"CROSS — Nbls: {uvd_combined.Nbls}  Ntimes: {uvd_combined.Ntimes}  "
          f"Nfreqs: {uvd_combined.Nfreqs}")
    print(f"AUTOS — Nbls: {uvd_autos.Nbls}  Ntimes: {uvd_autos.Ntimes}  "
          f"Nfreqs: {uvd_autos.Nfreqs}")
else:
    missing = []
    if not cross_outfile.exists():
        missing.append(str(cross_outfile))
    if not autos_outfile.exists():
        missing.append(str(autos_outfile))
    raise FileNotFoundError(
        "Concatenated file(s) missing — run concat_uvh5_fast.py first:\n"
        + "\n".join(missing)
    )


###############################################################################
#  ____  _____ _     _____ ____ _____   ____  _     ____                      #
# / ___|| ____| |   | ____/ ___|_   _| | __ )| |   / ___|                    #
# \___ \|  _| | |   |  _|| |     | |   |  _ \| |   \___ \                    #
#  ___) | |___| |___| |__| |___  | |   | |_) | |___ ___) |                   #
# |____/|_____|_____|_____\____| |_|   |____/|_____|____/                    #
#                                                                             #
#  Re-derive the redundant-baseline group from the loaded data.               #
###############################################################################

pol = ['ee', 'nn']

red_bls, lens, angs = hp.utils.get_reds(
    uvd_meta,
    bl_error_tol=REDUNDANT_GROUP_BL_ERROR_TOL_DATA,
    add_autos=True,
    bl_len_range=REDUNDANT_GROUP_LEN_RANGE,
    bl_deg_range=REDUNDANT_GROUP_ANG_RANGE,
    pick_data_ants=True,
)

print("\nAll redundant groups:")
print("-" * 55)
for length, angle, bl_group in zip(lens, angs, red_bls):
    print(f"{length:.2f} m, {angle:.2f}°, {len(bl_group)} baselines")
print("-" * 55)

selection_index = None
for val, (ilen, iang) in enumerate(zip(lens, angs)):
    if np.abs(ilen - bl_len) <= 5 and np.abs(iang - bl_ang) <= 5:
        selection_index = val
        break

print(f"selection_index: {selection_index}")

redgrp0 = red_bls[int(selection_index)]
print(f"Selected 0° group: {redgrp0}")

red_bls_dat = uvd_meta.get_antpairs()
red_bls_int = [x for x in redgrp0 if x in red_bls_dat]
print(f"Intersection with data: {red_bls_int}")

redgrp_pol = [ii + (jj,) for ii in red_bls_int for jj in pol]
redgrp_unpol = [ii for ii in red_bls_int]
redgrp_pol_comb = list(itertools.chain.from_iterable([redgrp_pol]))
redgrp_unpol_comb = list(itertools.chain.from_iterable([redgrp_unpol]))
print(f"redgrp_unpol_comb ({len(redgrp_unpol_comb)}): {redgrp_unpol_comb}")

# SPW range (single window spanning all loaded channels)
spw_ranges = SPW_RANGES
if spw_ranges is None:
    spw_ranges = [(0, fch_max - fch_min)]
print(f"spw_ranges: {spw_ranges}")


###############################################################################
#  _   _ _____ _     ____  _____ ____    _____ _   _ _   _  ____              #
# | | | | ____| |   |  _ \| ____|  _ \  |  ___| | | | \ | |/ ___|            #
# | |_| |  _| | |   | |_) |  _| | |_) | | |_  | | | |  \| | |               #
# |  _  | |___| |___|  __/| |___|  _ <  |  _| | |_| | |\  | |___            #
# |_| |_|_____|_____|_|   |_____|_| \_\ |_|    \___/|_| \_|\____|            #
#                                                                             #
###############################################################################

def zero_fraction_nsample(uvd):
    """Count and fraction of zero entries in nsample_array."""
    arr = np.asarray(uvd.nsample_array)
    total = arr.size
    zeros = int((arr == 0).sum())
    return zeros, total, zeros / total if total else 0.0


def summarise_uvd_quality(uvd):
    """NaN / flag fraction summary for a UVData object."""
    summary = {}
    if hasattr(uvd, 'data_array'):
        arr = uvd.data_array
        summary['data_array'] = (np.isnan(arr).sum(), arr.size)
    if hasattr(uvd, 'nsample_array'):
        arr = uvd.nsample_array
        summary['nsample_array'] = (np.isnan(arr).sum(), arr.size)
    for attr in ('flag_array', 'flags', 'flags_array'):
        if hasattr(uvd, attr):
            arr = getattr(uvd, attr)
            summary[attr] = (np.count_nonzero(arr), arr.size)
            break
    for name, (count, total) in summary.items():
        pct = 100 * count / total if total else 0
        label = "flagged" if "flag" in name else "NaNs"
        print(f"  {name}: {count}/{total} {label} ({pct:.2f}%)")
    return summary


def summarise_nan_stats(uvp, max_stats=3, max_spws=2, max_indices=5):
    """Truncated NaN summary for stats_array inside a UVPSpec."""
    if not hasattr(uvp, 'stats_array'):
        print("No stats_array.")
        return
    total_nans, total_pts = 0, 0
    for stat, spw_dict in uvp.stats_array.items():
        for arr in spw_dict.values():
            total_nans += np.isnan(arr).sum()
            total_pts += arr.size
    pct = 100 * total_nans / total_pts if total_pts else 0
    print(f"Overall NaNs in stats_array: {total_nans}/{total_pts} ({pct:.2f}%)")


def stats_nan_fraction_per_blpair(uvp):
    """Per-blpair NaN% in stats_array."""
    if not hasattr(uvp, 'stats_array'):
        return {}
    blpairs = np.array(uvp.blpair_array)
    nan_c = np.zeros(len(blpairs), dtype=int)
    tot_c = np.zeros(len(blpairs), dtype=int)
    for stat, spw_dict in uvp.stats_array.items():
        for arr in spw_dict.values():
            arr = np.asarray(arr)
            flat = arr.reshape(arr.shape[0], -1)
            nan_c += np.isnan(flat).sum(axis=1)
            tot_c += flat.shape[1]
    return dict(zip(blpairs.tolist(), (100 * nan_c / tot_c).tolist()))


def data_nan_fraction_per_blpair(uvp):
    """Per-blpair NaN% in data_array."""
    if not hasattr(uvp, 'data_array'):
        return {}
    blpairs = np.array(uvp.blpair_array)
    nan_c = np.zeros(len(blpairs), dtype=int)
    tot_c = np.zeros(len(blpairs), dtype=int)
    for spw, arr in uvp.data_array.items():
        arr = np.asarray(arr)
        if arr.shape[0] != len(blpairs):
            match_axes = [i for i, s in enumerate(arr.shape) if s == len(blpairs)]
            if match_axes:
                arr = np.moveaxis(arr, match_axes[0], 0)
        flat = arr.reshape(arr.shape[0], -1)
        nan_c += np.isnan(flat).sum(axis=1)
        tot_c += flat.shape[1]
    return dict(zip(blpairs.tolist(), (100 * nan_c / tot_c).tolist()))


###############################################################################
#  _   _       _   _   ____  _____ ____  _        _    ____ _____             #
# | \ | | __ _| \ | | |  _ \| ____|  _ \| |      / \  / ___| ____|           #
# |  \| |/ _` |  \| | | |_) |  _| | |_) | |     / _ \| |   |  _|            #
# | |\  | (_| | |\  | |  _ <| |___|  __/| |___ / ___ \ |___| |___           #
# |_| \_|\__,_|_| \_| |_| \_\_____|_|   |_____/_/   \_\____|_____|          #
#                                                                             #
#  Replace NaN visibilities with 0, smooth nsamples per spw.                  #
###############################################################################

print("\n--- Quality before NaN replacement ---")
summarise_uvd_quality(uvd_combined)
summarise_uvd_quality(uvd_autos)


def replace_nan_with_zero(uvd):
    uvd_out = deepcopy(uvd)
    uvd_out.data_array = np.nan_to_num(uvd_out.data_array, nan=0)
    return uvd_out


uvd_cleaned_inp = replace_nan_with_zero(uvd_combined)
uvd_cleaned_inp_auto = replace_nan_with_zero(uvd_autos)

print("\n--- Quality after NaN replacement ---")
summarise_uvd_quality(uvd_cleaned_inp)
summarise_uvd_quality(uvd_cleaned_inp_auto)
print(f"redgrp_unpol_comb: {redgrp_unpol_comb}")


###############################################################################
#  _   _ ____    _    __  __ ____  _     _____   ____  __  __  ___   ___ _____ #
# | \ | / ___|  / \  |  \/  |  _ \| |   | ____| / ___||  \/  |/ _ \ / _ \_   _|#
# |  \| \___ \ / _ \ | |\/| | |_) | |   |  _|   \___ \| |\/| | | | | | | || | #
# | |\  |___) / ___ \| |  | |  __/| |___| |___   ___) | |  | | |_| | |_| || | #
# |_| \_|____/_/   \_\_|  |_|_|   |_____|_____| |____/|_|  |_|\___/ \___/ |_| #
#                                                                             #
#  Replace per-channel nsamples with per-SPW mean for spectral smoothness.    #
###############################################################################

def replace_nsamples_with_median(uvd):
    uvd_out = deepcopy(uvd)
    for bl in uvd.get_antpairs():
        for pol_name in uvd.get_pols():
            pol_idx = uvd.get_pols().index(pol_name)
            key = bl + (pol_name,)
            nsamples = np.array(uvd.get_nsamples(key), copy=True)

            for index, spw_range in enumerate(spw_ranges):
                s0, s1 = spw_ranges[index]
                for t in range(nsamples.shape[0]):
                    temp = nsamples[t, s0:s1]
                    mean_value = np.mean(temp)
                    nsamples[t, s0:s1] = mean_value

                bl_idx = uvd.antpair2ind(bl)
                uvd_out.nsample_array[
                    bl_idx, s0:s1, pol_idx
                ] = nsamples[:, s0:s1]

    return uvd_out


uvd_cleaned_inp_nsmp = replace_nsamples_with_median(uvd_cleaned_inp)
uvd_cleaned_inp_nsmp_auto = replace_nsamples_with_median(uvd_cleaned_inp_auto)

print("\n--- Quality after nsample smoothing ---")
summarise_uvd_quality(uvd_cleaned_inp_nsmp)
summarise_uvd_quality(uvd_cleaned_inp_nsmp_auto)


###############################################################################
#  _____ __  __ ____ ___ ____  _____   _     _____        __                  #
# | ____|\ \/ // ___|_ _/ ___|| ____| | |   / _ \ \      / /                 #
# |  _|   \  /| |    | |\___ \|  _|   | |  | | | \ \ /\ / /                  #
# | |___  /  \| |___ | | ___) | |___  | |__| |_| |\ V  V /                   #
# |_____|/_/\_\\____|___|____/|_____| |_____\___/  \_/\_/                    #
#                                                                             #
#  Drop baselines whose max(nsample) == 0 for a given pol.                    #
###############################################################################

def excise_low_nsample_per_pol(uvd_x, uvd_auto=None, thresh=1):
    keep_keys = []
    for ant1, ant2, pol_name in uvd_x.get_antpairpols():
        ns = uvd_x.get_nsamples((ant1, ant2, pol_name))
        if np.nanmax(ns) > thresh:
            keep_keys.append((ant1, ant2, pol_name))

    new_x = deepcopy(uvd_x)
    if keep_keys:
        new_x.select(bls=keep_keys, keep_all_metadata=False, inplace=True)
    else:
        raise RuntimeError("All cross keys rejected by nsample threshold.")

    new_auto = None
    if uvd_auto is not None:
        survived = {}
        for a, b, p in keep_keys:
            survived.setdefault(a, set()).add(p)
            survived.setdefault(b, set()).add(p)

        keep_auto = [
            (a, a2, p)
            for a, a2, p in uvd_auto.get_antpairpols()
            if p in survived.get(a, set())
        ]
        new_auto = deepcopy(uvd_auto)
        if keep_auto:
            new_auto.select(bls=keep_auto, keep_all_metadata=False, inplace=True)
        else:
            raise RuntimeError("All autos dropped by nsample criterion.")

    return new_x, new_auto


if cut_switch == 1:
    print("\nExcising low-nsample baselines...")
    uvd_cleaned_inp_nsmp_cut, uvd_cleaned_inp_nsmp_auto_cut = \
        excise_low_nsample_per_pol(
            uvd_cleaned_inp_nsmp, uvd_cleaned_inp_nsmp_auto, thresh=0
        )
else:
    print("\nSkipping baseline excision.")
    uvd_cleaned_inp_nsmp_cut = deepcopy(uvd_cleaned_inp_nsmp)
    uvd_cleaned_inp_nsmp_auto_cut = deepcopy(uvd_cleaned_inp_nsmp_auto)

redgrp_unpol_comb_cut = uvd_cleaned_inp_nsmp_cut.get_antpairs()
autos_set_cut = uvd_cleaned_inp_nsmp_auto_cut.get_antpairs()

print(f"Cross bls remaining: {len(uvd_cleaned_inp_nsmp_cut.get_antpairpols())}")
print(f"Auto  bls remaining: {len(uvd_cleaned_inp_nsmp_auto_cut.get_antpairpols())}")


###############################################################################
#  ____  _____ ____    _    __     ______                                     #
# |  _ \| ____|  _ \  / \   \ \   / / ___|                                   #
# | |_) |  _| | | | |/ _ \   \ \ / / |  _                                    #
# |  _ <| |___| |_| / ___ \   \ V /| |_| |                                   #
# |_| \_\_____|____/_/   \_\   \_/  \____|                                   #
#                                                                             #
#  Redundant-average visibilities (coherent average in visibility space).      #
###############################################################################

red_avg_uvd = utils.red_average(uvd_cleaned_inp_nsmp_cut, propagate_flags=True)

print("\n--- Quality of redundant-averaged UVD ---")
summarise_uvd_quality(red_avg_uvd)
show_ram()


###############################################################################
#   ____ ___  _   _ _____ ____  _____ _   _ _____                             #
#  / ___/ _ \| | | | ____|  _ \| ____| \ | |_   _|                           #
# | |  | | | | |_| |  _| | |_) |  _| |  \| | | |                            #
# | |__| |_| |  _  | |___|  _ <| |___| |\  | | |                            #
#  \____\___/|_| |_|_____|_| \_\_____|_| \_| |_|                            #
#                                                                             #
#  ____  ____  _____ ____                                                     #
# |  _ \/ ___||  _ \| ____|  ___                                              #
# | |_) \___ \| |_) |  _|  / __|                                             #
# |  __/ ___) |  __/| |___| (__                                               #
# |_|   |____/|_|   |_____|\___| (red-avg'd data × itself)                   #
#                                                                             #
###############################################################################

print("\n--- Coherent PSPEC ---")
print(f"spw_ranges: {spw_ranges}")

cosmo = hp.conversions.Cosmo_Conversions()
polpair_in = (pol_in, pol_in)

# Load beam
beam = UVBeam()
beam.read_beamfits(beamfile)
print(f"Beam pols: {beam.polarization_array}")
print(f"Data pols: {red_avg_uvd.polarization_array}")

uvb = hp.pspecbeam.PSpecBeamUV(beam, cosmo=cosmo)

output_file_coh = os.path.join(save_path_scratch, f"Junk_{batchnum}_sim.h5")

ds = pspecdata.pspec_run(
    dsets=[red_avg_uvd],
    filename=output_file_coh,
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
    broadcast_dset_flags=True,
)

ds.Jy_to_mK()
ds.rephase_to_dset(0)
ds.dsets[0].vis_units = 'mK'

baselines = red_avg_uvd.get_antpairs()
print(f"Coherent PSPEC baselines: {baselines}")

uvp = ds.pspec(
    baselines, baselines, (0, 0),
    [polpair_in],
    spw_ranges=spw_ranges,
    input_data_weight=PSPEC_INPUT_DATA_WEIGHT,
    norm=PSPEC_NORM,
    taper=PSPEC_TAPER,
    verbose=True,
)

print(f"Delays: {len(uvp.get_dlys(0))} bins")
print(f"LSTs:   {len(np.unique(uvp.lst_avg_array))} unique")

# Clean up temp file
if os.path.isfile(output_file_coh):
    os.remove(output_file_coh)
    print(f"Deleted temp file: {output_file_coh}")

show_ram()

if FREE_INTERMEDIATE_OBJECTS:
    free_named_objects("red_avg_uvd", "ds")


###############################################################################
#  ___ _   _  ____ ___  _   _ _____ ____  _____ _   _ _____                   #
# |_ _| \ | |/ ___/ _ \| | | | ____|  _ \| ____| \ | |_   _|                #
#  | ||  \| | |  | | | | |_| |  _| | |_) |  _| |  \| | | |                  #
#  | || |\  | |__| |_| |  _  | |___|  _ <| |___| |\  | | |                  #
# |___|_| \_|\____\___/|_| |_|_____|_| \_\_____|_| \_| |_|                  #
#                                                                             #
#  ____  ____  _____ ____                                                     #
# |  _ \/ ___||  _ \| ____|  ___                                              #
# | |_) \___ \| |_) |  _|  / __|                                             #
# |  __/ ___) |  __/| |___| (__                                               #
# |_|   |____/|_|   |_____|\___| (per-baseline, then average in P-space)     #
#                                                                             #
###############################################################################

print("\n--- Incoherent PSPEC ---")

uvd_cleaned_inp_all = uvd_cleaned_inp_nsmp_cut + uvd_cleaned_inp_nsmp_auto_cut

output_file_inc = os.path.join(save_path_scratch, f"Junk_{batchnum}.h5")

dsinc = pspecdata.pspec_run(
    dsets=[uvd_cleaned_inp_all],
    filename=output_file_inc,
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
    broadcast_dset_flags=True,
    exclude_cross_bls=True,
    store_cov=STORE_INCOHERENT_COVARIANCE,
    store_cov_diag=STORE_INCOHERENT_COVARIANCE_DIAG,
    cov_model=INCOHERENT_COV_MODEL,
)

dsinc.Jy_to_mK()
dsinc.rephase_to_dset(0)
dsinc.dsets[0].vis_units = 'mK'

baselines_inc = uvd_cleaned_inp_nsmp_cut.get_antpairs()

uvpinc = dsinc.pspec(
    baselines_inc, baselines_inc, (0, 0),
    [polpair_in],
    spw_ranges=spw_ranges,
    input_data_weight=PSPEC_INPUT_DATA_WEIGHT,
    norm=PSPEC_NORM,
    taper=PSPEC_TAPER,
    verbose=True,
    store_cov=STORE_INCOHERENT_COVARIANCE,
    store_cov_diag=STORE_INCOHERENT_COVARIANCE_DIAG,
    cov_model=INCOHERENT_COV_MODEL,
)

print(f"Incoherent UVPSpec: {uvpinc}")

# Clean up temp file
if os.path.isfile(output_file_inc):
    os.remove(output_file_inc)
    print(f"Deleted temp file: {output_file_inc}")

show_ram()

if FREE_INTERMEDIATE_OBJECTS:
    free_named_objects(
        "uvd_combined",
        "uvd_autos",
        "uvd_cleaned_inp",
        "uvd_cleaned_inp_auto",
        "uvd_cleaned_inp_nsmp",
        "uvd_cleaned_inp_nsmp_auto",
        "uvd_cleaned_inp_nsmp_cut",
        "uvd_cleaned_inp_nsmp_auto_cut",
        "uvd_cleaned_inp_all",
        "dsinc",
        "beam",
        "uvb",
    )


###############################################################################
#  ____  _   _   _   _       _   _    _    _   _ ____  _     ___ _   _  ____  #
# |  _ \| \ | | | \ | | __ _| \ | |  | | | | / \| \ | |  _ \| |   |_ _| \ | |/ ___|#
# | |_) |  \| | |  \| |/ _` |  \| |  | |_| |/ _ \|  \| || | | | |    | ||  \| | |  _ #
# |  __/| |\  | | |\  | (_| | |\  |  |  _  / ___ \ |\  || |_| | |___ | || |\  | |_| |#
# |_|   |_| \_| |_| \_|\__,_|_| \_|  |_| |_/_/   \_\_| \_||____/|_____|___|_| \_|\____|#
#                                                                             #
#  Replace NaNs in the incoherent UVPSpec arrays with zeros.                  #
###############################################################################

def compute_nan_fraction_uvp(uvp_obj):
    summary = {}
    for attr in ['data_array', 'integration_array', 'wgt_array', 'nsample_array']:
        if hasattr(uvp_obj, attr):
            arr_dict = getattr(uvp_obj, attr)
            total = sum(a.size for a in arr_dict.values())
            nans = sum(np.isnan(a).sum() for a in arr_dict.values())
            summary[attr] = (nans, total, nans / total if total else 0.0)
    if hasattr(uvp_obj, 'stats_array'):
        total, nans = 0, 0
        for sd in uvp_obj.stats_array.values():
            for a in sd.values():
                total += a.size
                nans += np.isnan(a).sum()
        summary['stats_array'] = (nans, total, nans / total if total else 0.0)
    return summary


def replace_nan_with_zero_uvp(uvp_obj, inplace=False):
    summary_before = compute_nan_fraction_uvp(uvp_obj)
    uvp_clean = uvp_obj if inplace else deepcopy(uvp_obj)

    def _clean(d):
        for k, a in d.items():
            d[k] = np.nan_to_num(a, nan=0.0)

    for attr in ['data_array', 'integration_array', 'wgt_array', 'nsample_array']:
        if hasattr(uvp_clean, attr):
            _clean(getattr(uvp_clean, attr))
    if hasattr(uvp_clean, 'stats_array'):
        for stat, sd in uvp_clean.stats_array.items():
            _clean(sd)

    summary_after = compute_nan_fraction_uvp(uvp_clean)
    return uvp_clean, summary_before, summary_after


print("\n--- Cleaning NaNs in incoherent UVPSpec ---")
uvpinc, before, after = replace_nan_with_zero_uvp(
    uvpinc, inplace=CLEAN_UVPINC_INPLACE
)
for name in before:
    b_n, b_t, b_f = before[name]
    a_n, a_t, a_f = after[name]
    print(f"  {name}: {b_n}/{b_t} ({b_f:.2%}) -> {a_n}/{a_t} ({a_f:.2%})")

gc.collect()
malloc_trim()


###############################################################################
#  ___ _   _  ____ ___  _   _    _    ____                                    #
# |_ _| \ | |/ ___/ _ \| | | |  / \  / ___|                                  #
#  | ||  \| | |  | | | | |_| | / _ \| |                                      #
#  | || |\  | |__| |_| |  _  |/ ___ \ |___                                   #
# |___|_| \_|\____\___/|_| |_/_/   \_\____|                                  #
#                                                                             #
#     _   __     _______ ____      _    ____ _____                            #
#    / \  \ \   / / ____|  _ \    / \  / ___| ____|                           #
#   / _ \  \ \ / /|  _| | |_) |  / _ \| |  _|  _|                            #
#  / ___ \  \ V / | |___|  _ <  / ___ \ |_| | |___                           #
# /_/   \_\  \_/  |_____|_| \_\/_/   \_\____|_____|                          #
#                                                                             #
#  Incoherent baseline-pair average in power-spectrum space.                   #
#  Uses a local copy of hera_pspec.average_spectra with diagnostic prints.     #
###############################################################################

def average_spectra(uvp_in, blpair_groups=None, time_avg=False,
                    blpair_weights=None, error_field=None,
                    error_weights=None, normalize_weights=True,
                    inplace=True, add_to_history=''):
    """
    Average power spectra across the baseline-pair-time axis, weighted by
    each spectrum's integration time or a specified kind of error bars.

    This is an "incoherent" average: it averages power spectra, not
    visibility data.
    """
    if inplace:
        uvp = uvp_in
    else:
        uvp = deepcopy(uvp_in)

    blpair_groups = deepcopy(blpair_groups)
    blpair_weights = deepcopy(blpair_weights)

    if blpair_groups is not None:
        assert isinstance(blpair_groups[0], (list, np.ndarray))
        if isinstance(blpair_groups[0][0], tuple):
            blpair_groups = [
                [uvp.antnums_to_blpair(blp) for blp in blpg]
                for blpg in blpair_groups
            ]
        uvp_blpairs = [uvp.antnums_to_blpair(blp) for blp in uvp.get_blpairs()]
        blvecs_groups = []
        for group in blpair_groups:
            blvecs_groups.append(
                uvp.get_blpair_blvecs()[uvp_blpairs.index(group[0])]
            )
        blpair_lens, _ = pspec_utils.get_bl_lens_angs(
            blvecs_groups, bl_error_tol=1.
        )
    else:
        _, idx = np.unique(uvp.blpair_array, return_index=True)
        blpair_groups = [[blp] for blp in uvp.blpair_array[np.sort(idx)]]
        blpair_lens = [blv for blv in uvp.get_blpair_seps()[np.sort(idx)]]
        assert blpair_weights is None

    all_blpairs = [item for sub in blpair_groups for item in sub]
    if len(set(all_blpairs)) < len(all_blpairs):
        print("Warning: some baseline-pairs repeated between groups.")

    if blpair_weights is None:
        blpair_weights = [[1. for _ in grp] for grp in blpair_groups]

    use_error_weights = error_weights is not None
    if use_error_weights and error_weights not in uvp.stats_array:
        raise KeyError(f"error_field '{error_weights}' not in stats_array.")

    stat_l = []
    if isinstance(error_field, (list, tuple, np.ndarray)):
        stat_l = list(error_field)
    elif isinstance(error_field, str):
        stat_l = [error_field]
    if use_error_weights and error_weights not in stat_l:
        stat_l.append(error_weights)

    if not uvp.exact_windows:
        extra = set(uvp.blpair_array) - set(all_blpairs)
        blpair_groups += [[blp] for blp in extra]
        blpair_weights += [[1.] for _ in extra]

    data_array, wgts_array = odict(), odict()
    ints_array, nsmp_array = odict(), odict()
    stats_array = odict([[s, odict()] for s in stat_l])
    store_cov = hasattr(uvp, "cov_array_real")
    if store_cov:
        cov_array_real, cov_array_imag = odict(), odict()
    store_window = hasattr(uvp, 'window_function_array')
    if store_window:
        window_function_array = odict()
        window_function_kperp, window_function_kpara = odict(), odict()

    for spw in range(uvp.Nspws):
        spw_data, spw_wgts, spw_ints, spw_nsmp = [], [], [], []
        spw_stats = odict([[s, []] for s in stat_l])
        if store_window:
            spw_wf = []
            spw_wf_kp, spw_wf_kpa = [], []
        if store_cov:
            spw_cr, spw_ci = [], []

        for i, p in enumerate(uvp.polpair_array):
            pol_data, pol_wgts, pol_ints, pol_nsmp = [], [], [], []
            pol_stats = odict([[s, []] for s in stat_l])
            if store_window:
                pol_wf = []
            if store_cov:
                pol_cr, pol_ci = [], []

            for j, blpg in enumerate(blpair_groups):
                bpg_data, bpg_wgts, bpg_ints, bpg_nsmp = [], [], [], []
                bpg_stats = odict([[s, []] for s in stat_l])
                if store_window:
                    bpg_wf = []
                if store_cov:
                    bpg_cr, bpg_ci = [], []
                w_list = []

                blpg_wgts = np.array(blpair_weights[j])
                norm_val = np.sum(blpg_wgts) if normalize_weights else 1.
                if norm_val <= 0.:
                    raise ValueError(f"Sum of weights in group {j} <= 0.")
                blpg_wgts = blpg_wgts * float(blpg_wgts.size) / norm_val

                for k, blp in enumerate(blpg):
                    nsmp = uvp.get_nsamples((spw, blp, p))[:, None]
                    data = uvp.get_data((spw, blp, p))
                    wgts = uvp.get_wgts((spw, blp, p))
                    ints = uvp.get_integrations((spw, blp, p))[:, None]
                    if store_window:
                        wf = uvp.get_window_function((spw, blp, p))
                    if store_cov:
                        cr = uvp.get_cov((spw, blp, p), component="real")
                        ci = uvp.get_cov((spw, blp, p), component="imag")

                    errws = {}
                    for s in stat_l:
                        errws[s] = uvp.get_stats(s, (spw, blp, p)).copy()
                        np.square(errws[s], out=errws[s],
                                  where=np.isfinite(errws[s]))

                    if use_error_weights:
                        sv = uvp.get_stats(error_weights, (spw, blp, p)).copy().real
                        np.square(sv, out=sv, where=np.isfinite(sv))
                        sv = np.nan_to_num(sv, nan=np.inf, posinf=np.inf)
                        w = np.real(1. / sv.clip(1e-40, np.inf))
                    else:
                        w = (ints * np.sqrt(nsmp))**2

                    if time_avg:
                        wsum = np.sum(w, axis=0).clip(1e-40, np.inf)
                        data = (np.sum(data * w, axis=0) / wsum)[None]
                        wgts = (np.sum(wgts * w[:, :1, None], axis=0) / wsum[:1, None])[None]
                        ints = (np.sum(ints * w, axis=0) / wsum)[None]
                        nsmp = np.sum(nsmp, axis=0)[None]
                        if store_window:
                            if uvp.exact_windows:
                                wf = (np.sum(wf * w[:, :, None, None], axis=0) / wsum[:, None, None])[None]
                            else:
                                wf = (np.sum(wf * w[:, :, None], axis=0) / wsum[:, None])[None]
                        if store_cov:
                            cr = (np.sum(cr * w[:, :, None] * w[:, None, :], axis=0) / wsum[:, None] / wsum[None, :])[None]
                            ci = (np.sum(ci * w[:, :, None] * w[:, None, :], axis=0) / wsum[:, None] / wsum[None, :])[None]
                        for s in stat_l:
                            we = errws[s].clip(0, 1e40) * w**2
                            errws[s] = (np.sum(we, axis=0) / wsum**2)[None]
                            errws[s][np.isclose(errws[s], 0)] = np.inf
                        w = np.sum(w, axis=0)[None]

                    for m in range(int(blpg_wgts[k])):
                        bpg_data.append(data * w)
                        bpg_wgts.append(wgts * w[:, :1, None])
                        bpg_ints.append(ints * w)
                        bpg_nsmp.append(nsmp)
                        for s in stat_l:
                            bpg_stats[s].append(errws[s].clip(0, 1e40) * w**2)
                        if store_window:
                            if uvp.exact_windows:
                                bpg_wf.append(wf * w[:, :, None, None])
                            else:
                                bpg_wf.append(wf * w[:, :, None])
                        if store_cov:
                            bpg_cr.append(cr * w[:, :, None] * w[:, None, :])
                            bpg_ci.append(ci * w[:, :, None] * w[:, None, :])
                        w_list.append(w)

                wls = np.sum(w_list, axis=0).clip(1e-40, np.inf)
                bpg_data = np.sum(bpg_data, axis=0) / wls
                bpg_wgts = np.sum(bpg_wgts, axis=0) / wls[:, :1, None]
                bpg_nsmp = np.sum(bpg_nsmp, axis=0)
                bpg_ints = np.sum(bpg_ints, axis=0) / wls
                if store_cov:
                    bpg_cr = np.sum(bpg_cr, axis=0) / wls[:, :, None] / wls[:, None, :]
                    bpg_ci = np.sum(bpg_ci, axis=0) / wls[:, :, None] / wls[:, None, :]
                for s in stat_l:
                    sa = np.sum(bpg_stats[s], axis=0) / wls**2
                    sa[np.isclose(sa, 0)] = np.inf
                    bpg_stats[s] = np.sqrt(sa)
                if store_window:
                    if uvp.exact_windows:
                        bpg_wf = np.sum(bpg_wf, axis=0)
                    else:
                        bpg_wf = np.sum(bpg_wf, axis=0) / wls[:, :, None]

                pol_data.extend(bpg_data)
                pol_wgts.extend(bpg_wgts)
                pol_ints.extend(bpg_ints)
                pol_nsmp.extend(bpg_nsmp)
                for s in stat_l:
                    pol_stats[s].extend(bpg_stats[s])
                if store_window:
                    pol_wf.extend(bpg_wf)
                if store_cov:
                    pol_cr.extend(bpg_cr)
                    pol_ci.extend(bpg_ci)

            spw_data.append(pol_data)
            spw_wgts.append(pol_wgts)
            spw_ints.append(pol_ints)
            spw_nsmp.append(pol_nsmp)
            for s in stat_l:
                spw_stats[s].append(pol_stats[s])
            if store_window:
                spw_wf.append(pol_wf)
                if uvp.exact_windows:
                    spw_wf_kp.append(uvp.window_function_kperp[spw][:, i])
                    spw_wf_kpa.append(uvp.window_function_kpara[spw][:, i])
            if store_cov:
                spw_cr.append(pol_cr)
                spw_ci.append(pol_ci)

        data_array[spw] = np.moveaxis(spw_data, 0, -1)
        wgts_array[spw] = np.moveaxis(spw_wgts, 0, -1)
        ints_array[spw] = np.moveaxis(spw_ints, 0, -1)[:, 0, :]
        nsmp_array[spw] = np.moveaxis(spw_nsmp, 0, -1)[:, 0, :]
        for s in stat_l:
            stats_array[s][spw] = np.moveaxis(spw_stats[s], 0, -1)
        if store_window:
            window_function_array[spw] = np.moveaxis(spw_wf, 0, -1)
            if uvp.exact_windows:
                window_function_kperp[spw] = np.moveaxis(spw_wf_kp, 0, -1)
                window_function_kpara[spw] = np.moveaxis(spw_wf_kpa, 0, -1)
        if store_cov:
            cov_array_real[spw] = np.moveaxis(np.array(spw_cr), 0, -1)
            cov_array_imag[spw] = np.moveaxis(np.array(spw_ci), 0, -1)

    # Metadata
    time_1, time_2, time_avg_arr = [], [], []
    lst_1, lst_2, lst_avg_arr = [], [], []
    blpair_arr, bl_arr = [], []

    for i, blpg in enumerate(blpair_groups):
        blpairts = uvp.blpair_to_indices(blpg[0])
        bl_arr.extend(list(uvputils._blpair_to_bls(blpg[0])))
        if time_avg:
            blpair_arr.append(blpg[0])
            time_1.extend([np.mean(uvp.time_1_array[blpairts])])
            time_2.extend([np.mean(uvp.time_2_array[blpairts])])
            time_avg_arr.extend([np.mean(uvp.time_avg_array[blpairts])])
            lst_1.extend([np.mean(np.unwrap(uvp.lst_1_array[blpairts])) % (2*np.pi)])
            lst_2.extend([np.mean(np.unwrap(uvp.lst_2_array[blpairts])) % (2*np.pi)])
            lst_avg_arr.extend([np.mean(np.unwrap(uvp.lst_avg_array[blpairts])) % (2*np.pi)])
        else:
            blpair_arr.extend(np.ones_like(blpairts, int) * blpg[0])
            time_1.extend(uvp.time_1_array[blpairts])
            time_2.extend(uvp.time_2_array[blpairts])
            time_avg_arr.extend(uvp.time_avg_array[blpairts])
            lst_1.extend(uvp.lst_1_array[blpairts])
            lst_2.extend(uvp.lst_2_array[blpairts])
            lst_avg_arr.extend(uvp.lst_avg_array[blpairts])

    bl_arr = np.array(sorted(set(bl_arr)))
    bl_vecs = np.array([
        uvp.bl_vecs[uvp.bl_array.tolist().index(bl)] for bl in bl_arr
    ])

    uvp.Ntimes = len(np.unique(np.hstack([time_1, time_2])))
    uvp.Nbltpairs = len(time_avg_arr)
    uvp.Nblpairs = len(np.unique(blpair_arr))
    uvp.Nbls = len(bl_arr)
    uvp.Ntpairs = len(set((t1, t2) for t1, t2 in zip(time_1, time_2)))
    uvp.bl_array = bl_arr
    uvp.bl_vecs = bl_vecs
    uvp.blpair_array = np.array(blpair_arr)
    uvp.time_1_array = np.array(time_1)
    uvp.time_2_array = np.array(time_2)
    uvp.time_avg_array = np.array(time_avg_arr)
    uvp.lst_1_array = np.array(lst_1)
    uvp.lst_2_array = np.array(lst_2)
    uvp.lst_avg_array = np.array(lst_avg_arr)
    uvp.data_array = data_array
    uvp.integration_array = ints_array
    uvp.wgt_array = wgts_array
    uvp.nsample_array = nsmp_array
    if store_window:
        uvp.window_function_array = window_function_array
        if uvp.exact_windows:
            uvp.window_function_kperp = window_function_kperp
            uvp.window_function_kpara = window_function_kpara
    if store_cov:
        uvp.cov_array_real = cov_array_real
        uvp.cov_array_imag = cov_array_imag
    if stat_l:
        uvp.stats_array = stats_array
    elif hasattr(uvp, "stats_array"):
        delattr(uvp, "stats_array")

    uvp.history = (
        f"Spectra averaged with hera_pspec\n{add_to_history}\n"
        + "-" * 40 + "\n" + uvp.history
    )
    uvp.check()

    return uvp


# ── Run the incoherent average ───────────────────────────────────────────────
# For perfectly redundant visibilities with equal weights, the coherent average
# reduces noise before squaring while the incoherent average keeps a larger
# noise bias:
#
#   P_inc(τ) - P_coh(τ) ≃ (1 - 1/N_red) P_N(τ).
#
# Deviations from this toy expectation are the signal-loss / non-redundancy
# effects this script is designed to expose.

all_blpairs = uvpinc.get_blpairs()
blpair_groups = [all_blpairs]
print(f"\nAveraging {len(all_blpairs)} baseline-pairs incoherently...")

uvpspec_averaged = average_spectra(
    uvp_in=uvpinc,
    blpair_groups=blpair_groups,
    error_field=INCOHERENT_AVERAGE_ERROR_FIELD,
    time_avg=INCOHERENT_AVERAGE_TIME,
    inplace=AVERAGE_SPECTRA_INPLACE,
)

print(f"Averaged UVPSpec: {uvpspec_averaged}")
summarise_nan_stats(uvpspec_averaged)


###############################################################################
#  ____    _  __     _______ ____  ____   ___  ____  _   _  ____ _____ ____   #
# / ___|  / \\ \   / / ____|  _ \|  _ \ / _ \|  _ \| | | |/ ___|_   _/ ___|  #
# \___ \ / _ \\ \ / /|  _| | |_) | |_) | | | | | | | | | | |     | | \___ \  #
#  ___) / ___ \\ V / | |___|  __/|  _ <| |_| | |_| | |_| | |___  | |  ___) | #
# |____/_/   \_\\_/  |_____|_|   |_| \_\\___/|____/ \___/ \____| |_| |____/  #
#                                                                             #
#  Write coherent + incoherent PSPEC to HDF5 via PSpecContainer.              #
###############################################################################

m_ideal = re.search(r"subset_([^_]+)", base)
ideal_tag = m_ideal.group(1) if m_ideal else "ideal"
m_airy = re.search(r"nonred_([^_]+)$", base)
airy_tag = m_airy.group(1) if m_airy else "airy"

print(f"\nideal_tag: {ideal_tag}")
print(f"airy_tag : {airy_tag}")
print(f"sky_type : {sky_type}")
print(f"DATA_PATH: {DATA_PATH}")

# Coherent PSPEC
output_coh = os.path.join(
    str(DATA_PATH),
    f"notebook_{sky_type}_{ideal_tag}_{airy_tag}_Coh_PSPEC_"
    f"{bl_len}_{bl_ang}_{pol_in}_ck{chunk_min:05d}-{chunk_max:05d}_"
    f"{PROC_BASELINE_LABEL}_{PROC_LST_LABEL}_bch_{batchnum}_psc_PN.h5"
)
print(f"Saving coherent PSPEC:   {output_coh}")
psc = hp.PSpecContainer(output_coh, mode='rw', keep_open=False)
psc.set_pspec('dset0', 'dset0_x_dset0', uvp, overwrite=True)

# Incoherent PSPEC
output_inc = os.path.join(
    str(DATA_PATH),
    f"notebook_{sky_type}_{ideal_tag}_{airy_tag}_Incoh_PSPEC_"
    f"{bl_len}_{bl_ang}_{pol_in}_ck{chunk_min:05d}-{chunk_max:05d}_"
    f"{PROC_BASELINE_LABEL}_{PROC_LST_LABEL}_bch_{batchnum}_psc_PN.h5"
)
print(f"Saving incoherent PSPEC: {output_inc}")
psc = hp.PSpecContainer(output_inc, mode='rw', keep_open=False)
psc.set_pspec('dset0', 'dset0_x_dset0', uvpspec_averaged, overwrite=True)

show_ram()


###############################################################################
print("\n" + "=" * 70)
print("CODE HAS FINISHED")
print("=" * 70)
