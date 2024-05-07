#!/usr/bin/env python3
"""Utilities for running a full simulation."""
import logging
import subprocess

import hera_sim
import matvis

from core.obsparams import make_hera_obsparam

from . import utils
from ._cli_utils import _get_sbatch_program

logger = logging.getLogger(__name__)


def calculate_expected_time(chunks: int) -> int:
    """
    Calculate the expected time for a given number of chunks.

    Calibrated to Bridges GPU.
    """
    nt = 17280 // chunks

    # 5 min buffer + (nt / 5760) * 1 hr
    return int(max(5, (nt / 5760) * 60))


def run_validation_sim(
    layout: str,
    channels: list[int],
    sky_model: str,
    n_time_chunks: int,
    gpu: bool,
    slurm_override,
    skip_existing: bool,
    force_remake_obsparams: bool,
    log_level: str,
    dry_run: bool,
    simulator: str = "matvis",
    freq_interp_kind: str = "cubic",
    spline_interp_order: int = 1,
    do_time_chunks: list[int] | None = None,
    profile: bool = False,
):
    """Run a full validation sim on SLURM compute."""
    sgpu = "gpu" if gpu else "cpu"
    simulator_config = utils.REPODIR / f"{simulator}-{sgpu}.yaml"

    assert (
        simulator_config.exists()
    ), f"Simulator config file {simulator_config.name} does not exist."

    logger.info(f"Frequency channels to run: {channels}")

    layout_file = make_hera_obsparam(
        layout=layout,
        channels=channels,
        sky_model=sky_model,
        chunks=n_time_chunks,
        freq_interp_kind=freq_interp_kind,
        spline_interp_order=spline_interp_order,
        force=force_remake_obsparams,
        do_chunks=do_time_chunks,
    )

    if not do_time_chunks:
        do_time_chunks = list(range(n_time_chunks))

    time_est = calculate_expected_time(n_time_chunks)

    # We want to override the job-name to be <sky_model>-<fch>-<ch>, but the last two
    # variables have to be accessed in the loop, so we will be instead override it to
    # a Python string formatting pattern and format it in the loop.
    # Note that click default `slurm_overrride` to (), and we want it to be "2D" tuple
    logfile = utils.DIRFMT.format(
        sky_model=sky_model, chunks=n_time_chunks, layout=layout_file.stem
    )
    logdir = utils.LOGDIR / "vis" / logfile

    logdir.mkdir(parents=True, exist_ok=True)
    slurm_override = slurm_override + (
        ("job-name", "{sky_model}-fch{fch:04d}-chunk{ch}"),
        ("output", "{logdir}/fch{fch:04d}-ch{ch:03d}_%J.out"),
    )

    if "time" not in [x[0] for x in slurm_override]:
        slurm_override = slurm_override + (("time", f"0-00:{time_est}:00"),)

    # Make the SBATCH script minus hera-sim-vis.py command
    program = _get_sbatch_program(gpu, slurm_override)

    out_dir = utils.OUTDIR / utils.VIS_DIRFMT.format(
        sky_model=sky_model, chunks=n_time_chunks, layout=layout_file.stem
    )
    obsp_dir = utils.OBSPDIR / utils.OBSPARAM_DIRFMT.format(
        sky_model=sky_model, chunks=n_time_chunks, layout=layout_file.stem
    )

    compress_cache = utils.COMPRESSDIR / utils.COMPRESS_FMT.format(
        chunks=n_time_chunks, layout_file=layout_file.stem
    )
    if not utils.COMPRESSDIR.exists():
        utils.COMPRESSDIR.mkdir(parents=True)

    # Option for hera-sim-vis.py. Let's just keep this fixed.
    sim_options = (
        f"--normalize_beams --fix_autos --compress {compress_cache} "
        f"--log-level {log_level}"
    )

    for fch in channels:
        for ch in do_time_chunks:
            logger.info(f"Working on frequency channel {fch} chunk {ch}")

            outfile = out_dir / "{}.uvh5".format(
                utils.VIS_FLFMT.format(
                    sky_model=sky_model, fch=fch, ch=ch, layout=layout_file.stem
                )
            )
            obsp = obsp_dir / utils.OBSPARAM_FLFMT.format(
                sky_model=sky_model, ch=ch, fch=fch, layout=layout_file.stem
            )
            logger.info(f"{outfile.as_posix()}")
            # Check if output file already existed, if clobber is False
            if skip_existing and outfile.exists():
                logger.warning(f"File {outfile} exists, skipping")
                continue

            trace = " "
            profilestr = ""
            if profile:
                proflabel = (
                    f"{sky_model}-gpu{gpu}-nt{n_time_chunks}-{layout_file.stem}-"
                    f"mv{matvis.__version__}-hs{hera_sim.__version__}"
                )
                if gpu:
                    trace = (
                        "nsys profile -w true -t cuda,cublas -s cpu -f true -x true "
                        f"-o profiles/{proflabel} "
                    )
                profilestr = (
                    f"--profile --profile-output profiling/{proflabel}.profile.txt"
                )
                prof_funcs = [
                    "hera_sim.visibilities.simulators:VisibilitySimulation",
                    "hera_sim.visibilities.simulators:ModelData",
                ]
                if simulator == "matvis":
                    prof_funcs.extend(
                        [
                            f"matvis.{'gpu' if gpu else 'cpu'}:simulate",
                            "hera_sim.visibilities.matvis:MatVis",
                        ]
                    )
                elif simulator == "fftvis":
                    prof_funcs.extend(
                        [
                            "fftvis.simulate:simulate",
                            "hera_sim.visibilities.fftvis:FFTVis",
                        ]
                    )

                prof_funcs = ",".join(prof_funcs)
                profilestr += f' --profile-funcs "{prof_funcs}"'

            cmd = (
                f"{trace}hera-sim-vis.py {sim_options} {profilestr} {obsp} "
                f"{simulator_config}"
            )

            if utils.HPC_CONFIG["slurm"]:
                # Write job script and submit
                sbatch_dir = utils.REPODIR / "batch_scripts/vis"
                sbatch_dir.mkdir(parents=True, exist_ok=True)
                sbatch_file = sbatch_dir / utils.FLFMT.format(
                    sky_model=sky_model, fch=fch, ch=ch, layout=layout_file.stem
                )

                logger.info(f"Creating sbatch file: {sbatch_file}")
                # Now, join the job script with the hera-sim-vis.py command
                # and format the job-name
                job_script = "\n".join([program, "", cmd, ""]).format(
                    sky_model=sky_model, fch=fch, ch=ch, logdir=logdir
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
