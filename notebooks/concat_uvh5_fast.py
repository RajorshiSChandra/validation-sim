from pathlib import Path
import re
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyuvdata import UVData
import hera_pspec as hp

# ------------------------------------------------------------------
# HOW TO RUN :
# nohup python notebooks/concat_uvh5_fast.py > FILENAME.log 2>&1 &
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------------

BASE_INDIR = Path("/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/outputs")
BASE_OUTDIR = Path("/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/outputs")

# PTSRC SKY =============================================

# Airy######################
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

# Vivaldi###################
# ideal ENU, ideal Vivaldi       vivaldired, idealT
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
# real ENU, ideal Vivaldi       vivaldired, idealF
# MODEL_DIR   = Path("ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")


# EOR SKY =============================================

# Airy######################
# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")

# ideal ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealT
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")
# real ENU, tilt Airy var (deltZa ~ 2-3 degree)       airytilt, idealF
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airytilt")

# Vivaldi###################
# ideal ENU, ideal Vivaldi       vivaldired, idealT
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")
# real ENU, ideal Vivaldi       vivaldired, idealF
# MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_vivaldired")

# Gaussian###################
# MODEL_DIR  = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_gauss_skysd111_eoroffsetfix.03001951._beammapperant_airyred-nonred_airyred/")


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
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_556_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")    
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR  = Path("eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/fftvis_xcheck/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")
    # Rlzn 556, 14m Airy, CV rlzn challenge
# MODEL_DIR  = Path("eor-grf-256/seed700_freqslic_middle_fch0273ref/fftvis_xcheck/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D14m_airy_nonspectral_80MHzref_eoroffsetfix.7be63f46._beammapperant_airyred-nonred_airyred/")
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
# MODEL_DIR   = Path("eor-grf-256-noisy-2x/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_masky_models/raw/p_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-correct-beam/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-D7m-airy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy.5cfe1b35._beammapperant_airyred-nonred_airyred/")      #seed 777
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


DATA_PATH = BASE_INDIR / MODEL_DIR
OUT_PATH  = BASE_OUTDIR / MODEL_DIR
print("DATA_PATH (input) :", DATA_PATH)
print("OUT_PATH  (output):", OUT_PATH)

t_global_start = time.time()

fch_min, fch_max = 271, 276          # inclusive channel range      # 227
chunk_min, chunk_max = 0, 288         # inclusive chunk range    # 96

bl_len = 29.0                        # target baseline length [m]
bl_ang = 0.0                         # target angle [deg], e.g. EW group

len_tol = 2.0                        # ± length tolerance [m]
ang_tol = 5.0                        # ± angle tolerance [deg]

pol_in = "xx"                         # one of {"xx", "yy", "xy", "yx"}

N_WORKERS = 8                         # number of parallel file readers

# ------------------------------------------------------------------
# HELPER: build filename for a given (channel, chunk)
# ------------------------------------------------------------------

def fname_for(ch, chunk):
    """
    Return full path to fch####_chunk#####.uvh5 for given channel, chunk.
    """
    return DATA_PATH / f"fch{ch:04d}_chunk{chunk:05d}.uvh5"

# ------------------------------------------------------------------
# STEP 1: Build a metadata UVData to define redundant group
# ------------------------------------------------------------------

ref_ch = fch_min
ref_ck = chunk_min
ref_fn = fname_for(ref_ch, ref_ck)
if not ref_fn.exists():
    raise FileNotFoundError(f"Reference file {ref_fn} not found.")

uvd_meta = UVData()
uvd_meta.read_uvh5(ref_fn, read_data=False)
print(f"Loaded metadata from: {ref_fn}")
print("Nants_data:", uvd_meta.Nants_data, "Nbls:", uvd_meta.Nbls)

# ------------------------------------------------------------------
# STEP 2: Find redundant group near (bl_len, bl_ang)
# ------------------------------------------------------------------

red_bls, lens, angs = hp.utils.get_reds(
    uvd_meta,
    bl_error_tol=2.0,
    add_autos=True,
    bl_len_range=(10.0, 100.0),
    bl_deg_range=(0.0, 180.0),
    pick_data_ants=True,
)

print("Found", len(red_bls), "redundant groups.")

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

redgrp = red_bls[best_idx]
print("Number of baselines in group (incl. autos if present):", len(redgrp))
print("redgrp:", redgrp)

redgrp_unpol = [(a1, a2) for (a1, a2) in redgrp if a1 != a2]
print("Cross-only baselines in group:", len(redgrp_unpol))

autos_set = sorted({(a, a) for (a1, a2) in redgrp for a in (a1, a2)})
print("Associated autos:", autos_set)

# ------------------------------------------------------------------
# STEP 3: Read files in parallel, single read per file
# ------------------------------------------------------------------

# Combine cross + auto baselines into one list for a single read per file
all_bls = redgrp_unpol + autos_set
auto_antpairs_set = set(autos_set)

def read_one_file(args):
    """
    Read a single uvh5 file for all needed baselines (cross + auto),
    then split into cross and auto UVData objects.
    Returns (ch, uv_cross, uv_auto) or None if file doesn't exist.
    """
    fn, ch = args
    if not fn.exists():
        return None

    uv = UVData()
    uv.read_uvh5(fn, bls=all_bls, polarizations=[pol_in])

    # Split into cross and auto
    uv_cross = uv.select(bls=redgrp_unpol, inplace=False)
    uv_auto  = uv.select(bls=autos_set, inplace=False)

    return (ch, uv_cross, uv_auto)


# Build list of all (filename, channel) pairs
fnames = []
channels = range(fch_min, fch_max + 1)
chunks   = range(chunk_min, chunk_max + 1)
for ch in channels:
    for ck in chunks:
        fn = fname_for(ch, ck)
        fnames.append((fn, ch))

total_files = len(fnames)
print(f"\nTotal files to process: {total_files}")
print(f"Using {N_WORKERS} parallel workers...")

from collections import defaultdict
cross_by_ch = defaultdict(list)   # ch -> [UVData, ...]
auto_by_ch  = defaultdict(list)

n_read = 0
n_missing = 0
t_start = time.time()

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    future_to_fn = {pool.submit(read_one_file, item): item for item in fnames}

    for future in as_completed(future_to_fn):
        item = future_to_fn[future]
        try:
            result = future.result()
        except Exception as exc:
            print(f"[ERROR] {item[0]} generated exception: {exc}")
            continue

        if result is None:
            n_missing += 1
            continue

        ch, uv_cross, uv_auto = result
        cross_by_ch[ch].append(uv_cross)
        auto_by_ch[ch].append(uv_auto)
        n_read += 1

        if n_read % 100 == 0:
            elapsed = time.time() - t_start
            rate = n_read / elapsed
            remaining = (total_files - n_read - n_missing) / rate if rate > 0 else 0
            print(f"  ... read {n_read}/{total_files} files "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                  f"{rate:.1f} files/s)")

t_read = time.time() - t_start
print(f"\nFiles read: {n_read}, missing/skipped: {n_missing}")
print(f"Read phase took {t_read:.1f}s ({n_read/t_read:.1f} files/s)")

if len(cross_by_ch) == 0:
    raise RuntimeError("No cross-correlation data was read.")
if len(auto_by_ch) == 0:
    raise RuntimeError("No autocorrelation data was read.")

# ------------------------------------------------------------------
# STEP 4: Two-stage fast concatenation
#   Stage 1: concat chunks within each channel along "blt" (same freq)
#   Stage 2: concat across channels along "freq"
# ------------------------------------------------------------------

print(f"\nStage 1: Concatenating chunks within each of {len(cross_by_ch)} channels...")
t_concat1 = time.time()

cross_per_ch = []
auto_per_ch  = []
n_ch = len(sorted(cross_by_ch.keys()))
for i, ch in enumerate(sorted(cross_by_ch.keys())):
    clist = cross_by_ch[ch]
    if len(clist) == 1:
        cross_per_ch.append(clist[0])
    else:
        cross_per_ch.append(clist[0].fast_concat(clist[1:], axis="blt", inplace=False))

    alist = auto_by_ch[ch]
    if len(alist) == 1:
        auto_per_ch.append(alist[0])
    else:
        auto_per_ch.append(alist[0].fast_concat(alist[1:], axis="blt", inplace=False))

    # Sort by time so lst_array is consistent across channels
    cross_per_ch[-1].reorder_blts(order="time")
    auto_per_ch[-1].reorder_blts(order="time")

    if (i + 1) % 10 == 0 or (i + 1) == n_ch:
        elapsed = time.time() - t_concat1
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (n_ch - i - 1) / rate if rate > 0 else 0
        print(f"  ... concat channel {i+1}/{n_ch} "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

# Free per-chunk memory
del cross_by_ch, auto_by_ch
print(f"Stage 1 took {time.time() - t_concat1:.1f}s")

print(f"\nStage 2: Concatenating across {len(cross_per_ch)} channels...")
t_concat2 = time.time()

if len(cross_per_ch) == 1:
    uvd_combined = cross_per_ch[0]
else:
    uvd_combined = cross_per_ch[0].fast_concat(cross_per_ch[1:], axis="freq", inplace=False)
del cross_per_ch
print(f"  Cross freq-concat done ({time.time() - t_concat2:.1f}s)")

t_auto2 = time.time()
if len(auto_per_ch) == 1:
    uvd_autos = auto_per_ch[0]
else:
    uvd_autos = auto_per_ch[0].fast_concat(auto_per_ch[1:], axis="freq", inplace=False)
del auto_per_ch
print(f"  Auto freq-concat done ({time.time() - t_auto2:.1f}s)")
print(f"Stage 2 took {time.time() - t_concat2:.1f}s")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print("\n=== Combined CROSS data ===")
print("Nbls:", uvd_combined.Nbls, "Ntimes:", uvd_combined.Ntimes, "Nfreqs:", uvd_combined.Nfreqs)
print("Unique antpairs (cross):", uvd_combined.get_antpairs())
print("Polarizations:", uvd_combined.polarization_array)

print("\n=== Combined AUTOS data ===")
print("Nbls:", uvd_autos.Nbls, "Ntimes:", uvd_autos.Ntimes, "Nfreqs:", uvd_autos.Nfreqs)
print("Unique antpairs (autos):", uvd_autos.get_antpairs())
print("Polarizations:", uvd_autos.polarization_array)

lst_hours_cross = np.unique(uvd_combined.lst_array) * (12.0 / np.pi)
lst_hours_autos = np.unique(uvd_autos.lst_array) * (12.0 / np.pi)
print("\nLSTs (cross) [hr]:", lst_hours_cross)
print("LSTs (autos) [hr]:", lst_hours_autos)

# ------------------------------------------------------------------
# STEP 5: Write outputs
# ------------------------------------------------------------------

OUT_PATH.mkdir(parents=True, exist_ok=True)

base = MODEL_DIR.name
m_ideal = re.search(r"subset_([^_]+)_", base)
ideal_tag = m_ideal.group(1) if m_ideal else "ideal"

m_airy = re.search(r"nonred_([^_]+)$", base)
airy_tag = m_airy.group(1) if m_airy else "airy"

print("ideal_tag:", ideal_tag)
print("airy_tag :", airy_tag)

tag_stem = (
    f"{ideal_tag}_{airy_tag}"
    f"_fch{fch_min:04d}-{fch_max:04d}"
    f"_ck{chunk_min:05d}-{chunk_max:05d}"
    f"_bl{bl_len:.1f}m"
)

cross_outfile = OUT_PATH / f"uvd_cross_{tag_stem}.uvh5"
autos_outfile = OUT_PATH / f"uvd_autos_{tag_stem}.uvh5"

print("Cross output file:", cross_outfile)
print("Autos output file:", autos_outfile)

t_write = time.time()
uvd_combined.write_uvh5(cross_outfile, clobber=True)
print(f"  Cross written ({time.time() - t_write:.1f}s)")

t_w2 = time.time()
uvd_autos.write_uvh5(autos_outfile, clobber=True)
print(f"  Autos written ({time.time() - t_w2:.1f}s)")

t_total = time.time() - t_global_start
print(f"\nAll done! Total time: {t_total:.1f}s ({t_total/60:.1f} min)")