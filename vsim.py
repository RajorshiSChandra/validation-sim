#!/usr/bin/env python3
"""Entry point for top-level CLI."""
import logging
import subprocess
from pathlib import Path

import click
from rich.logging import RichHandler

from core import _cli_utils as _cli
from core import sky_model as sm
from core import utils
from core.obsparams import make_hera_obsparam
from core.run_sim import run_validation_sim

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}

logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Make job scripts and run visibility simulations via hera-sim-vis.py."""
    pass


@cli.command("runsim")
@_cli.opts.add_opts
def hera_cli(channels, freq_range, **kwargs):
    """Run HERA validation simulations.

    Use the default parameters, configuration files, and directories for HERA sims
    (see make_obsparams.py).
    """
    channels = _cli.parse_channels(channels, freq_range)

    run_validation_sim(channels=channels, **kwargs)


@cli.command("make-obsparams")
@_cli.opts.layout
@_cli.opts.ants
@_cli.opts.channels
@_cli.opts.freq_range
@_cli.opts.sky_model
@_cli.opts.n_time_chunks
def make_obsparams(layout, freq_range, channels, sky_model, n_time_chunks):
    """Make obsparams for H4C simulations given a sky model and frequencies."""
    channels = _cli.parse_channels(channels, freq_range)

    make_hera_obsparam(
        layout=layout, channels=channels, sky_model=sky_model, chunks=n_time_chunks
    )


option_nside = click.option("--nside", default=256, show_default=True)


@cli.command("sky-model")
@click.argument("sky_model", type=click.Choice(["gsm", "diffuse", "ptsrc", "eor"]))
@_cli.opts.channels
@_cli.opts.freq_range
@_cli.opts.slurm_override
@_cli.opts.skip_existing
@_cli.opts.dry_run
@option_nside
@click.option("--local/--slurm", default=False)
@click.option("--split-freqs/--no-split-freqs", default=False)
def sky_model(
    sky_model,
    freq_range,
    channels,
    nside,
    local,
    slurm_override,
    split_freqs,
    skip_existing,
    dry_run,
):
    """Make SkyModel at given frequencies.

    Frequencies are based on H4C data.
    Outputs are written to the default directories, i.e. "./sky_models/<type>".
    """
    channels = _cli.parse_channels(channels, freq_range)
    if local:
        if sky_model == "gsm":
            sm.make_gsm_model(channels, nside)
        elif sky_model == "diffuse":
            sm.make_diffuse_model(channels, nside)
        elif sky_model == "ptsrc":
            sm.make_ptsrc_model(channels, nside)
        elif sky_model == "eor":
            raise NotImplementedError("eor sky model not supported yet")
    else:
        sm.run_make_sky_model(
            sky_model,
            channels,
            nside,
            slurm_override=slurm_override,
            skip_existing=skip_existing,
            dry_run=dry_run,
            split_freqs=split_freqs,
        )


@cli.command("cornerturn")
@_cli.opts.sky_model
@click.option("-c", "--time-chunk", default=0)
@click.option("-n", "--new-chunk-size", default=2)
@click.option("--nchunks-sim", default=3, type=int)
@click.option("--conjugate/--no-conjugate", default=False)
@click.option("--remove-cross-pols/--keep-cross-pols", default=False)
@click.option(
    "--direc",
    default=None,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--channels",
    default=None,
    type=str,
    help="Channels to use, e.g. '0~1536'. If not given, all channels are used.",
)
@_cli.opts.layout
@_cli.opts.log_level
@_cli.opts.dry_run
@_cli.opts.slurm_override
def cornerturn(
    sky_model,
    time_chunk,
    slurm_override,
    new_chunk_size,
    dry_run,
    nchunks_sim,
    conjugate: bool,
    remove_cross_pols: bool,
    direc: Path | None,
    channels: str | None,
    log_level: str,
    layout: str,
):
    """Perform a cornerturn on simulation files.

    This takes multiple files, each with a single frequency and many times (snapshots),
    and reforms them into files with all frequencies and a set number of times (generally
    smaller). Note that the input files may be partial in frequency *and* time.

    Output files have the following prototype:
        zen.LST.{lst:.7f}[.{sky_cmp}].uvh5
    """
    logger.setLevel(log_level)

    # Make sure that the slurm log directory exists.
    # Otherwise, the job will terminate
    log_dir = Path(f"logs/chunk/{sky_model}")
    log_dir.mkdir(parents=True, exist_ok=True)

    if direc is None:
        simdir = utils.OUTDIR / utils.VIS_DIRFMT.format(
            sky_model=sky_model, chunks=nchunks_sim, layout=layout
        )
    else:
        simdir = Path(direc)

    outdir = (
        utils.OUTDIR
        / utils.VIS_DIRFMT.format(
            sky_model=sky_model, chunks=nchunks_sim, layout=layout
        )
        / "rechunk"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    conjugate = "--conjugate" if conjugate else ""
    remove_cross_pols = "--remove-cross-pols" if remove_cross_pols else ""

    if channels is None:
        allfiles = sorted(
            simdir.glob(f"{sky_model}_fch????_nt17280_chunk{time_chunk}.uvh5")
        )
        maxchan = int(allfiles[-1].name.split("fch")[1][:4])
        if len(allfiles) != maxchan + 1:
            raise ValueError(f"Missing files in {simdir}")
        channels = f"0~{maxchan+1}"

    nchannels = int(channels.split("~")[1]) - int(channels.split("~")[0])
    estimated_time = 36 * nchannels / 1536  # hours

    estimated_minutes = max(int(estimated_time - int(estimated_time)) * 60, 10)

    if estimated_time > 24:
        estimated_time = f"1-{int(estimated_time)-24:02d}:{estimated_minutes:02d}:00"
    else:
        estimated_time = f"{int(estimated_time):02d}:{estimated_minutes:02d}:00"

    slurm_override = slurm_override + (
        ("job-name", f"{sky_model}-ct"),
        ("output", f"{log_dir}/%J.out"),
        ("nodes", "1"),
        ("ntasks", "1"),
        ("cpus-per-task", "16"),
        ("mem", "16GB"),
        ("time", estimated_time),
    )

    sbatch = _cli._get_sbatch_program(gpu=False, slurm_override=slurm_override)

    cmd = f"""
    time python core/rechunk-fast.py \
    --r-prototype "{sky_model}_fch{{channel:04d}}_nt17280_chunk{time_chunk}.uvh5" \
    --chunk-size {new_chunk_size} \
    --channels {channels} \
    --sky-cmp {sky_model}\
    --assume-same-blt-layout \
    --is-rectangular \
    --nthreads 16 \
    {conjugate} \
    {remove_cross_pols} \
    --log-level {log_level} \
    {simdir} \
    {outdir} \
    """
    sbatch_dir = utils.REPODIR / "batch_scripts/rechunk"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    sbatch_file = sbatch_dir / f"{sky_model}_ch{time_chunk}.sbatch"

    sbatch = "\n".join([sbatch, "", cmd, ""])
    with open(sbatch_file, "w") as fl:
        fl.write(sbatch)

    if not dry_run:
        subprocess.call(f"sbatch {sbatch_file}".split())

    logger.debug(f"\n===Job Script===\n{sbatch}\n===END===\n")


if __name__ == "__main__":
    cli()
