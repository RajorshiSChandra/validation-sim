from pathlib import Path
import numpy as np
from pyuvdata import UVData
import hera_pspec as hp   # hp.utils.get_reds

# ------------------------------------------------------------------
# HOW TO RUN :
# nohup python notebooks/concat_uvh5.py > FILENAME.log 2>&1 &
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------------

BASE_OUTDIR= Path("/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/outputs") 
# / "ptsrc256/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyprb"
# / "ptsrc256/nt17280-00288chunks-HERA_custom_subset_cba81417555edaffd87557575713cb61.txt-nonred"

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
MODEL_DIR   = Path("eor-grf-256/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
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

# EOR Noisy =============================================

# ideal ENU, ideal Airy                               airyred, idealT
# MODEL_DIR   = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt-nonred_airyred")
# MODEL_DIR   = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-2x/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-correct-beam/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")
# MODEL_DIR   = Path("eor-grf-256-noisy-D7m-airy/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_D7m_airy.5cfe1b35._beammapperant_airyred-nonred_airyred/")
# real ENU, ideal Airy                                airyred, idealF
# MODEL_DIR   = Path("eor-grf-256-noisy/nt17280-00288chunks-HERA_custom_subset_idealF_cba81417555edaffd87557575713cb61.txt-nonred_airyred")

# PTSRC Noisy =============================================

# PURE Noise =============================================

# MODEL_DIR   = Path("noise-only-300k/nt17280-00288chunks-HERA_custom_subset_idealT_cba81417555edaffd87557575713cb61.txt.beam_map_10_airy.9105a7bf._beammapperant_airyred-nonred_airyred/")


DATA_PATH = BASE_OUTDIR / MODEL_DIR
print("DATA_PATH ", DATA_PATH)

fch_min, fch_max = 227, 315          # inclusive channel range
chunk_min, chunk_max = 0, 47         # inclusive chunk range

bl_len = 28.0                        # target baseline length [m]
bl_ang = 0.0                         # target angle [deg], e.g. EW group

len_tol = 2.0                       # ± length tolerance [m]
ang_tol = 5.0                        # ± angle tolerance [deg]

pol_in = "xx"                        # one of {"xx", "yy", "xy", "yx"}

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
# STEP 3: Loop over (fch, chunk), read only this group into UVData
# ------------------------------------------------------------------

uvd_combined = None   # cross-corr group
uvd_autos     = None   # autos group

channels = range(fch_min, fch_max + 1)
chunks   = range(chunk_min, chunk_max + 1)

for ch in channels:
    for ck in chunks:
        fn = fname_for(ch, ck)
        if not fn.exists():
            print(f"[WARN] Missing file {fn}, skipping.")
            continue

        # print(f"Reading {fn} for cross-corr group...")
        uv_tmp = UVData()
        uv_tmp.read_uvh5(fn, bls=redgrp_unpol, polarizations=[pol_in])

        if uvd_combined is None:
            uvd_combined = uv_tmp
        else:
            uvd_combined += uv_tmp

        # print(f"Reading {fn} for autos...")
        uv_auto_tmp = UVData()
        uv_auto_tmp.read_uvh5(fn, bls=autos_set, polarizations=[pol_in])

        if uvd_autos is None:
            uvd_autos = uv_auto_tmp
        else:
            uvd_autos += uv_auto_tmp

# ------------------------------------------------------------------
# outputs
# ------------------------------------------------------------------

if uvd_combined is None:
    raise RuntimeError("uvd_combined is empty, no files were successfully read.")
if uvd_autos is None:
    raise RuntimeError("uvd_autos is empty, no autos were successfully read.")

print("\n=== Combined CROSS data ===")
print("Nbls:", uvd_combined.Nbls, "Ntimes:", uvd_combined.Ntimes, "Nfreqs:", uvd_combined.Nfreqs)
print("Unique antpairs (cross):", uvd_combined.get_antpairs())
print("Polarizations:", uvd_combined.polarization_array)

print("\n=== Combined AUTOS data ===")
print("Nbls:", uvd_autos.Nbls, "Ntimes:", uvd_autos.Ntimes, "Nfreqs:", uvd_autos.Nfreqs)
print("Unique antpairs (autos):", uvd_autos.get_antpairs())
print("Polarizations:", uvd_autos.polarization_array)

# Optional: check LSTs in hours (your preferred convention)
lst_hours_cross = np.unique(uvd_combined.lst_array) * (12.0 / np.pi)
lst_hours_autos = np.unique(uvd_autos.lst_array) * (12.0 / np.pi)
print("\nLSTs (cross) [hr]:", lst_hours_cross)
print("LSTs (autos) [hr]:", lst_hours_autos)


# %%
import re

# output directory exists
DATA_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Parse tags from MODEL_DIR
# ------------------------------------------------------------------
base = MODEL_DIR.name
# "nt17280-00288chunks-HERA_custom_subset_idealF_cba814...txt-nonred_airytilt"

m_ideal = re.search(r"subset_([^_]+)_", base)
ideal_tag = m_ideal.group(1) if m_ideal else "ideal"

m_airy = re.search(r"nonred_([^_]+)$", base)
airy_tag = m_airy.group(1) if m_airy else "airy"

print("ideal_tag:", ideal_tag)
print("airy_tag :", airy_tag)

# ------------------------------------------------------------------
# Build filename stem with channel / chunk limits
# ------------------------------------------------------------------
tag_stem = (
    f"{ideal_tag}_{airy_tag}"
    f"_fch{fch_min:04d}-{fch_max:04d}"
    f"_ck{chunk_min:05d}-{chunk_max:05d}"
    f"_bl{bl_len:.1f}m"
)

cross_outfile = DATA_PATH / f"uvd_cross_{tag_stem}.uvh5"
autos_outfile = DATA_PATH / f"uvd_autos_{tag_stem}.uvh5"

print("Cross output file:", cross_outfile)
print("Autos output file:", autos_outfile)

# ------------------------------------------------------------------
# Write UVData objects to disk
# ------------------------------------------------------------------
uvd_combined.write_uvh5(cross_outfile, clobber=True)
uvd_autos.write_uvh5(autos_outfile, clobber=True)

print("Done writing uvd_combined and uvd_autos.")
