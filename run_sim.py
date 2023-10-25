#!/usr/bin/env python3
import subprocess
from pathlib import Path
import logging
import numpy as np
import yaml
import click
import utils
from make_obsparams import _make_hera_obsparam
from typing import Sequence

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=100)

logger = logging.getLogger(__name__)


# ===LOGIC===


def _get_sbatch_program(gpu: bool, slurm_override=None):
    conda_params = utils.HPC_CONFIG["conda"]
    module_params = utils.HPC_CONFIG["module"]
    slurm_params = utils.HPC_CONFIG["slurm"]['gpu' if gpu else 'cpu']

    if slurm_override is not None:  # Modify slurm options
        for k, v in slurm_override:
            slurm_params[k] = v

    shebang = "#!/bin/bash"
    sbatch = "\n".join([f"#SBATCH --{k}={v}" for k, v in slurm_params.items()])

    conda = """
source {conda_path}/bin/activate
conda activate {environment_name}
""".format_map(
        conda_params
    )

    module = "\n".join([f"module load {md}" for md in module_params])

    program = "\n".join([shebang, sbatch, conda, module])
    return program

def calculate_expected_time(chunks: int) -> int:
    nt = 17280 // chunks

    # 5 min buffer + (nt / 5760) * 1 hr
    return int(max(5, (nt / 5760) * 60))

def run_validation_sim(
    layout: str,
    freq_range: tuple[int, int],
    freqs: Sequence[int],
    sky_model: str,
    chunks: int,
    gpu: bool,
    slurm_override,
    skip_existing: bool,
    force_remake_obsparams: bool,
    log_level: str,
    dry_run: bool,
    freq_interp_kind: str = 'cubic', 
    spline_interp_order: int = 3,
    do_time_chunks: list[int] | None = None,
):
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, log_level)
    )

    out_dir = utils.OUTDIR / utils.VIS_DIRFMT.format(sky_model=sky_model, chunks=chunks)
    obsp_dir = utils.OBSPDIR / utils.OBSPARAM_DIRFMT.format(
        sky_model=sky_model, chunks=chunks
    )
    simulator_config = utils.REPODIR / "visgpu.yaml" if gpu else utils.REPODIR / "viscpu.yaml"

    if do_time_chunks is None:
        do_time_chunks = list(range(chunks))

    time_est = calculate_expected_time(chunks)

    # We want to override the job-name to be <sky_model>-<fch>-<ch>, but the last two
    # varaibles have to be accessed in the loop, so we will be instead override it to
    # a Python string formatting pattern and format it in the loop.
    # Note that click default `slurm_overrride` to (), and we want it to be "2D" tuple
    slurm_override = slurm_override + (
        ("job-name", "{sky_model}-fch{fch:04d}-chunk{ch}"),
        ("output", "logs/vis/{sky_model}/fch{fch:04d}-ch{ch:03d}_%J.out"),
        ("time", f"0-00:{time_est}:00"),
    )
    # Make the SBATCH script minus hera-sim-vis.py command
    program = _get_sbatch_program(gpu, slurm_override)

    freq_chans = np.arange(*freq_range)
    if freqs:
        extra_freqs = freqs[np.isin(freqs, freq_chans, invert=True)]
        freq_chans = np.append(freq_chans, extra_freqs)
    logger.info(f"Frequency channels to run: {freq_chans}")

    layout_file = _make_hera_obsparam(
        layout=layout, 
        freq_range=freq_range, 
        freqs=freqs, 
        sky_model=sky_model, 
        chunks=chunks, 
        freq_interp_kind = freq_interp_kind, 
        spline_interp_order = spline_interp_order, 
        force=force_remake_obsparams,
        do_chunks=do_time_chunks
    )

    compress_cache = utils.COMPRESSDIR / utils.COMPRESS_FMT.format(
        chunks=chunks, layout_file=layout_file.name
    )
    # Option for hera-sim-vis.py. Let's just keep this fixed.
    sim_options = (
        f"--normalize_beams --fix_autos --compress {compress_cache} "
        f"--log-level {log_level}"
    )

    for fch in freq_chans:
        for ch in do_time_chunks:
            logger.info(f"Working on frequency channel {fch} chunk {ch}")
            
            outfile = out_dir / utils.VIS_FLFMT.format(
                sky_model=sky_model, fch=fch, ch=ch
            )
            obsp = obsp_dir / utils.OBSPARAM_FLFMT.format(sky_model=sky_model, ch=ch, fch=fch)

            # Check if output file already existed, if clobber is False
            if skip_existing and outfile.exists():
                logger.warning(f"File {outfile} exists, skipping")
            else:
                cmd = f"hera-sim-vis.py {sim_options} {obsp} {simulator_config}"

                if utils.HPC_CONFIG["slurm"]:
                    # Write job script and submit
                    sbatch_dir = utils.REPODIR / "batch_scripts"
                    sbatch_dir.mkdir(parents=True, exist_ok=True)
                    sbatch_file = (
                        sbatch_dir / f"{sky_model}_fch{fch:04d}_chunk{ch:03d}.sbatch"
                    )
                    
                    logger.info(f"Creating sbatch file: {sbatch_file}")
                    # Now, join the job script with the hera-sim-vis.py command
                    # and format the job-name
                    job_script = "\n".join([program, "", cmd, ""]).format(
                        sky_model=sky_model, fch=fch, ch=ch
                    )
                    with open(sbatch_file, "w") as fl:
                        fl.write(job_script)

                    if not dry_run:
                        subprocess.call(f"sbatch {sbatch_file}".split())

                    logger.debug(f"\n===Job Script===\n{job_script}\n===END===\n")
                else:
                    logger.info(f"Running the simulation locally\nCommand: {cmd}")
                    subprocess.call(cmd.split()) if not dry_run else None


"""===CLI==="""


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Make job scripts and run visibility simulations via hera-sim-vis.py"""
    pass


@cli.command("hera")
@click.option(
    "--layout", default=None, type=click.Choice(list(utils.ANTS_DICT.keys())),
    help="A pre-defined HERA layout to use"
)
@click.option(
    "-a", "--ants", type=click.IntRange(0, 350), default=None, multiple=True,
    help="ants to use as a subset of the full HERA 350"
)
@click.option(
    "-fr",
    "--freq-range",
    nargs=2,
    default=(0, len(utils.FREQS_DICT['H4C'])),
    show_default=True,
    metavar="START STOP",
    type=click.IntRange(0, len(utils.FREQS_DICT['H4C'])),
    help="Frequency channel range (zero-based)",
)
@click.option(
    "-fch",
    "--freqs",
    multiple=True,
    default=None,
    type=click.IntRange(0, len(utils.FREQS_DICT['H4C'])),
    metavar="FREQ_CHAN",
    help="Frequency channel, allow multiple, add to --freq-range",
)
@click.option(
    "-sm",
    "--sky-model",
    default="ptsrc",
    show_default=True,
    type=click.Choice(["ptsrc", "diffuse_nside256", "eor"], case_sensitive=True),
    help="Sky model to simulate",
)
@click.option(
    "--n-time-chunks",
    default=3,
    show_default=True,
    help="Split the simulation into a number of time chunks",
)
@click.option(
    "--do-time-chunks",
    default=None,
    type=int,
    multiple=True,
    help="Only run the simulation for these time chunks (useful for debugging/exploring)"
)
@click.option("--gpu/--cpu", default=False, show_default=True, help="Use gpu or cpu")
@click.option(
    "-so",
    "--slurm-override",
    nargs=2,
    multiple=True,
    metavar="FLAG VALUE",
    help="Override slurm options in the hpc config (excluding job-name and output), "
    "allow multiple",
)
@click.option(
    "--skip-existing/--rerun-existing",
    default=True,
    show_default=True,
    help="Skip or rerun if the simulation output already exists",
)
@click.option(
    "--force-remake-obsparams",
    is_flag=True,
    help="If set, remake all obsparams before running the simulations",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["INFO", "WARNING", "ERROR", "DEBUG"], case_sensitive=True),
    show_default=True,
    help="Verbosity level, also pass the flag to hera-sim-vis.py",
)
@click.option("-d", "--dry-run", is_flag=True, help="Pass the flag to hera-sim-vis.py")
def hera_cli(
    layout,
    ants,
    freq_range,
    freqs,
    sky_model,
    n_time_chunks,
    do_time_chunks,
    gpu,
    slurm_override,
    skip_existing,
    force_remake_obsparams,
    log_level,
    dry_run,
):
    """Run HERA validation simulations.

    Use the default parameters, configuration files, and directories for HERA sims
    (see make_obsparams.py).
    """
    if layout is not None and ants is not None:
        raise ValueError("Provide only layout or ants")
    if layout is None and ants is None:
        raise ValueError("Either layout or ants must be provided")
    if ants is not None:
        layout = list(ants)

    run_validation_sim(
        layout=layout,
        freq_range=freq_range,
        freqs=freqs,
        sky_model=sky_model,
        chunks=n_time_chunks,
        do_time_chunks=do_time_chunks,
        gpu=gpu,
        slurm_override=slurm_override,
        skip_existing=skip_existing,
        force_remake_obsparams=force_remake_obsparams,
        log_level=log_level,
        dry_run=dry_run,
    )


@cli.command("custom")
def custom_cli():
    """Run custom simulations (not yet implemented).

    All configs to be provided.
    """
    pass


if __name__ == "__main__":
    cli()
