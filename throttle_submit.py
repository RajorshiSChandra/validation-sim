#!/usr/bin/env python3
"""
Throttled submission wrapper for vsim.py runsim.

Usage:
    ./throttle_submit.py --throttle 4 -- ./vsim.py runsim [all your normal flags]

What it does:
    1. Runs your vsim.py command with --dry-run appended (generates batch scripts)
    2. Finds the generated scripts from run_sim.py's stdout (modeldir)
    3. Builds a SLURM array wrapper that replays each script identically
    4. Submits the array job with %N throttle

Every job gets the exact same #SBATCH params, log paths, and hera-sim-vis.py
command as a direct submission — just rate-limited.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Absolute path to the validation-sim directory (where this script lives)
VSIM_DIR = Path(__file__).parent.resolve()


def _parse_range_args(vsim_args: list[str], flag: str, aliases: list[str] = None) -> list[str]:
    """Extract all values for a CLI flag from the raw arg list.

    Handles both '--flag value' and '--flag=value' styles, and
    collects repeated occurrences (Click's ``multiple=True``).
    Returns the raw string values (e.g. ['227~316', '400']).
    """
    all_flags = [flag] + (aliases or [])
    values = []
    it = iter(range(len(vsim_args)))
    for i in it:
        arg = vsim_args[i]
        for f in all_flags:
            if arg == f:
                # next token is the value
                if i + 1 < len(vsim_args):
                    values.append(vsim_args[i + 1])
                break
            elif arg.startswith(f + '='):
                values.append(arg.split('=', 1)[1])
                break
    return values


def _expand_int_ranges(raw_values: list[str]) -> list[int]:
    """Expand a list of raw CLI values ('227~316', '400') into sorted ints.

    Mirrors the IntRangeBuilder + combine_int_ranges logic in _cli_utils.
    """
    result = set()
    for v in raw_values:
        if '~' in v:
            lo, hi = v.split('~', 1)
            result.update(range(int(lo), int(hi)))
        else:
            result.add(int(v))
    return sorted(result)


def _get_expected_scripts(vsim_args: list[str]) -> set[str] | None:
    """Build the set of expected script basenames from CLI params.

    Returns None if the params couldn't be determined (falls back to glob-all).
    Filename format: fch{fch:04d}_chunk{ch:05d}  (from utils.FLFMT).
    """
    ch_raw = _parse_range_args(vsim_args, '--channels', ['-fch'])
    dtc_raw = _parse_range_args(vsim_args, '--do-time-chunks')

    if not ch_raw:
        return None  # can't filter without knowing channels

    channels = _expand_int_ranges(ch_raw)

    if dtc_raw:
        chunks = _expand_int_ranges(dtc_raw)
    else:
        # Fallback: --do-time-chunks not given → run_sim uses range(n_time_chunks)
        ntc_raw = _parse_range_args(vsim_args, '--n-time-chunks')
        if ntc_raw:
            chunks = list(range(int(ntc_raw[-1])))
        else:
            return None  # can't determine chunk range

    return {f"fch{fch:04d}_chunk{ch:05d}" for fch in channels for ch in chunks}


def main():
    # ─────────────────────────────────────────────────────────────
    # ARGUMENT PARSING
    # We use parse_known_args so that anything this script doesn't
    # recognize gets collected into vsim_args (the user's vsim command).
    # The '--' separator tells argparse where our flags end and
    # vsim's flags begin.
    # ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Throttled sbatch submission for vsim.py runsim",
        usage="%(prog)s --throttle N -- ./vsim.py runsim [flags]",
    )
    parser.add_argument(
        "--throttle", "-t", type=int, default=4,
        help="Max concurrent SLURM array tasks (default: 4)",
    )
    parser.add_argument(
        "--dry-run", "-d", action="store_true",
        help="Build the wrapper but don't actually sbatch it",
    )
    args, vsim_args = parser.parse_known_args()
    # args.throttle = 4 (or whatever the user passed)
    # args.dry_run  = True/False
    # vsim_args     = ['--', './vsim.py', 'runsim', '--channels', '227~316', ...]

    if not vsim_args:
        parser.error("Provide the vsim.py command after '--'")

    # Remove the '--' separator if it's the first element
    if vsim_args[0] == '--':
        vsim_args = vsim_args[1:]
    # vsim_args is now: ['./vsim.py', 'runsim', '--channels', '227~316', ...]

    # ─────────────────────────────────────────────────────────────
    # STEP 1: RUN VSIM.PY IN DRY-RUN MODE
    #
    # The --dry-run flag makes vsim.py's run_sim.py loop:
    #   for fch in channels:
    #       for ch in do_time_chunks:
    #           write sbatch script to batch_scripts/vis/<modeldir>/fch0227_chunk00005
    #           # but SKIP the actual subprocess.call("sbatch ...") 
    #
    # So we get all 4,272 scripts written to disk without submitting anything.
    # ─────────────────────────────────────────────────────────────
    if '--dry-run' not in vsim_args and '-d' not in vsim_args:
        vsim_args.append('--dry-run')  # force dry-run if user didn't include it

    print(f"[throttle] Running dry-run: {' '.join(vsim_args)}")

    # Run the vsim.py command as a subprocess, capturing its stdout/stderr
    result = subprocess.run(
        vsim_args,
        cwd=VSIM_DIR,       # run from the validation-sim directory
        capture_output=True, # capture stdout and stderr separately
        text=True,           # return strings instead of bytes
    )
    
    print(" Checkpoint B ")

    # Show the user what vsim.py printed (obsparams info, layout file, etc.)
    if result.stdout:
        print(" print(result.stdout) ")
        print(result.stdout)
        print(" print(result.stdout) DONE ")
    if result.stderr:
        print(" print(result.stderr, file=sys.stderr) ")
        print(result.stderr, file=sys.stderr)
        print(" print(result.stderr, file=sys.stderr) DONE ")
        
    print(" Checkpoint A ")

    # If vsim.py failed (e.g. missing config, bad args), stop here
    if result.returncode != 0:
        print(f"[throttle] ERROR: dry-run failed (exit code {result.returncode})")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────
    # STEP 2: FIND THE JOB-SPECIFIC SUBFOLDER
    #
    # run_sim.py prints "modeldir is <path>" to stdout, e.g.:
    #   modeldir is  eor-grf-256/rlzn_seed_222_offsetfix_freqslic_grf/nt17280-00288chunks-HERA_...
    #
    # This is the relative path under batch_scripts/vis/ where all
    # the fch*_chunk* scripts for THIS specific job were written.
    # We parse it from stdout so we don't have to reconstruct the
    # complex folder name ourselves.
    # ─────────────────────────────────────────────────────────────
    modeldir = None
    for line in result.stdout.splitlines():
        if line.startswith("modeldir is "):
            # Split on "modeldir is " and take everything after it
            modeldir = line.split("modeldir is ", 1)[1].strip()
            break

    if not modeldir:
        print("[throttle] ERROR: could not find 'modeldir is ...' in vsim output")
        sys.exit(1)

    # Full path to the job's batch script folder
    # e.g. /lustre/.../validation-sim/batch_scripts/vis/eor-grf-256/rlzn_.../nt17280-...
    job_dir = VSIM_DIR / "batch_scripts" / "vis" / modeldir
    print(f"[throttle] Job directory: {job_dir}")

    if not job_dir.exists():
        print(f"[throttle] ERROR: {job_dir} does not exist")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────
    # STEP 3: COLLECT ONLY THE SCRIPTS MATCHING THIS RUN'S PARAMS
    #
    # Each script is named like: fch0227_chunk00005 (no extension)
    # There's one per (frequency_channel, time_chunk) pair.
    # We parse --channels and --do-time-chunks from the CLI args to
    # build the expected set of filenames, so scripts from previous
    # runs (e.g. different chunk ranges) are excluded.
    # ─────────────────────────────────────────────────────────────
    expected = _get_expected_scripts(vsim_args)
    if expected is not None:
        scripts = sorted(
            s for s in job_dir.glob("fch*")
            if s.name in expected
        )
        print(f"[throttle] Filtered to {len(scripts)}/{len(list(job_dir.glob('fch*')))} "
              f"scripts matching CLI params ({len(expected)} expected)")
    else:
        print("[throttle] WARNING: could not parse channels/chunks from CLI args; "
              "collecting all scripts in job_dir")
        scripts = sorted(job_dir.glob("fch*"))  # fallback
    n_jobs = len(scripts)

    if n_jobs == 0:
        print(f"[throttle] ERROR: no fch* scripts found in {job_dir}")
        sys.exit(1)

    print(f"[throttle] Found {n_jobs} job scripts")

    # Write a text file listing the absolute path to each script, one per line.
    # The SLURM wrapper will use sed to pick line N for array task N.
    # e.g. line 1 = /lustre/.../fch0227_chunk00000
    #      line 2 = /lustre/.../fch0227_chunk00001
    #      ...
    joblist_path = job_dir / "vsim_job_list.txt"
    joblist_path.write_text("\n".join(str(s.resolve()) for s in scripts) + "\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: EXTRACT RESOURCE #SBATCH LINES FROM THE FIRST SCRIPT
    #
    # Each generated script has headers like:
    #   #SBATCH --partition=hera
    #   #SBATCH --mem=16GB
    #   #SBATCH --time=0-00:900:00
    #   #SBATCH --nice=100
    #   #SBATCH --job-name=...     (per-job, skip this)
    #   #SBATCH --output=...       (per-job, skip this)
    #
    # We copy the resource lines (partition, mem, time, nice, ntasks)
    # into our wrapper so the SLURM allocation is identical.
    # We skip job-name and output because those are different for each
    # task — we handle them at runtime inside the wrapper.
    # ─────────────────────────────────────────────────────────────
    first_text = scripts[0].read_text()
    resource_lines = []
    for line in first_text.splitlines():
        if not line.startswith("#SBATCH"):
            continue
        # Extract the key name (e.g. "partition" from "#SBATCH --partition=hera")
        key = line.split("=", 1)[0].replace("#SBATCH --", "").strip()
        # Skip per-job fields — we handle these at runtime for each array task
        if key in ("job-name", "output", "error"):
            continue
        resource_lines.append(line)

    # ─────────────────────────────────────────────────────────────
    # STEP 5: BUILD THE SLURM ARRAY WRAPPER SCRIPT
    #
    # This is the single .sbatch file we submit to SLURM.
    # It creates N array tasks (one per generated script).
    # SLURM runs at most %throttle tasks at a time.
    #
    # For each task:
    #   1. Read the script path from line N of the job list
    #   2. Extract the original --output path from that script
    #   3. Extract the original --job-name from that script
    #   4. Create the log directory
    #   5. Update this task's job name (so squeue shows the right name)
    #   6. Run the script via bash, redirecting output to the original log path
    #
    # Why bash and not sbatch?
    #   "bash script.sh" treats #SBATCH as comments — only the real
    #   commands run (conda activate, module load, hera-sim-vis.py).
    #   "sbatch script.sh" would create a NEW child job outside the
    #   array's %N throttle — defeating the whole purpose.
    # ─────────────────────────────────────────────────────────────
    wrapper = f"""#!/bin/bash
#SBATCH --job-name=vsim_throttled
#SBATCH --array=0-{n_jobs - 1}%{args.throttle}
{chr(10).join(resource_lines)}
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# ^^^ We send SLURM's default output to /dev/null because we redirect
#     stdout/stderr ourselves to the original per-job log path below.

cd {VSIM_DIR}

# Path to the file listing all job scripts (one absolute path per line)
JOBLIST={joblist_path}

# SLURM_ARRAY_TASK_ID is 0-indexed, sed line numbers are 1-indexed
LINE=$((SLURM_ARRAY_TASK_ID + 1))

# Pick the script for this array task (e.g. line 1 → fch0227_chunk00000)
SCRIPT=$(sed -n "${{LINE}}p" "$JOBLIST")

# ── Extract the original --output and --job-name from the inner script ──
# Each generated script has lines like:
#   #SBATCH --output=/lustre/.../logs/vis/.../fch0227_chunk00005-%J.out
#   #SBATCH --job-name=eor-grf-256/.../fch0227_chunk00005
# We grep for these and strip the prefix to get the value.
ORIG_OUTPUT=$(grep '#SBATCH --output=' "$SCRIPT" | head -1 | sed 's/#SBATCH --output=//')
ORIG_JOBNAME=$(grep '#SBATCH --job-name=' "$SCRIPT" | head -1 | sed 's/#SBATCH --job-name=//')

# The original output path contains %J which sbatch would replace with the
# job ID. Since we're running via bash (not sbatch), we do this replacement
# ourselves using SLURM_JOB_ID.
ORIG_OUTPUT="${{ORIG_OUTPUT//%J/$SLURM_JOB_ID}}"

# Create the log directory if it doesn't exist (same as sbatch would do)
mkdir -p "$(dirname "$ORIG_OUTPUT")"

# Update this array task's job name so `squeue` shows the original name
# instead of the generic "vsim_throttled". The 2>/dev/null || true means
# we silently ignore errors (e.g. if scontrol is unavailable).
scontrol update JobId=${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} JobName="$ORIG_JOBNAME" 2>/dev/null || true

# ── Run the generated script ──
# We wrap the echo header AND the bash call inside {{ ... }} so that
# everything — our debug header plus all of hera-sim-vis.py's output —
# lands in the same log file at the original --output path.
# "bash" treats the inner #SBATCH lines as comments (they start with #).
# Only the real commands execute: conda activate, module load, hera-sim-vis.py.
{{
echo "============================================"
echo "Array task   : ${{SLURM_ARRAY_TASK_ID}}"
echo "Script       : ${{SCRIPT}}"
echo "Log          : ${{ORIG_OUTPUT}}"
echo "Node         : $(hostname)"
echo "Date         : $(date)"
echo "============================================"
echo ""
echo "──── Script contents ────"
cat "${{SCRIPT}}"
echo "──── End script contents ────"
echo ""
bash "${{SCRIPT}}"
}} > "$ORIG_OUTPUT" 2>&1
"""

    # Write the wrapper to disk
    wrapper_path = job_dir / "vsim_throttled.sbatch"
    wrapper_path.write_text(wrapper)

    print(f"[throttle] Wrote wrapper: {wrapper_path}")
    print(f"[throttle] Array: 0-{n_jobs - 1} ({n_jobs} tasks, %{args.throttle} throttle)")

    # ─────────────────────────────────────────────────────────────
    # STEP 6: SUBMIT THE WRAPPER TO SLURM
    #
    # This single "sbatch" call creates all N array tasks.
    # SLURM queues them and runs at most %throttle at a time.
    # ─────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"[throttle] Dry-run mode — not submitting. To submit manually:")
        print(f"  sbatch {wrapper_path}")
    else:
        print(f"[throttle] Submitting...")
        subprocess.run(["sbatch", str(wrapper_path)], cwd=VSIM_DIR)


if __name__ == "__main__":
    main()
