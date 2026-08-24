#!/usr/bin/env python3
"""Entry point for top-level CLI."""
import logging
import subprocess
from pathlib import Path

import click
from rich.logging import RichHandler

from core import _cli_utils as _cli
from core import utils
from core.anabeam_config import build_analytic_beam_config
from core.obsparams import gate_beam_map_for_simulator
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


@cli.command
# @_cli.opts.layout
# @_cli.opts.ants
@_cli.opts.add_opts
@click.option("--simulator", type=click.Choice(["fftvis", "matvis", "fftvis64", "fftvis32", "matvis-cpu"]), default="matvis")
@click.option("--beam-map-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="CSV with columns: ant_number,beam_file (per-antenna beams; matvis only).")
@click.option("--beamvar-type", type=click.Choice(['vivaldired', 'airyred', 'airyprb', 'airytilt']), help="Type of beam variation to use for non-redundant sims.")
# Analytical beam options start
@click.option("--analytic-beam-class", type=click.Choice([
    "AiryBeam", "GaussianBeam", "hera_sim.beams.PolyBeam", 
    "hera_sim.beams.ZernikeBeam", "hera_sim.beams.PerturbedPolyBeam"
]), default=None, help="Analytical beam class to use")
@click.option("--analytic-beam-diameter", type=float, default=14.0, help="Diameter for AiryBeam (m)")
@click.option("--analytic-beam-sigma", type=float, default=0.15, help="Sigma for GaussianBeam (rad)")
@click.option("--analytic-beam-ref-freq", type=float, default=1.0e8, help="Reference freq (Hz)")
@click.option("--analytic-beam-spectral-index", type=float, default=-0.6975, help="Spectral index")
@click.option("--analytic-beam-coeffs-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--analytic-beam-preset", type=click.Choice(["fagnoni19"]), default=None)
@click.option("--analytic-beam-map-file", type=click.Path(exists=True, path_type=Path), default=None,
              help="YAML file mapping antennas to different analytical beam configs")
# Analytical beam options end
def runsim(layout, ants, channels, freq_range,
           simulator, beam_map_csv, beamvar_type,
           analytic_beam_class, analytic_beam_diameter, analytic_beam_sigma,
           analytic_beam_ref_freq, analytic_beam_spectral_index,
           analytic_beam_coeffs_file, analytic_beam_preset,
           analytic_beam_map_file,
           **kwargs):
    """Run HERA validation simulations.

    Use the default parameters, configuration files, and directories for HERA sims
    (see make_obsparams.py).
    """
    from core.run_sim import run_validation_sim
    
    # Resolve layout/ants precedence 
    if ants and layout:
        raise click.BadParameter("Do not provide both --layout and --ants")
    if ants:
        layout = ants
    if not ants and not layout:
        raise click.BadParameter("You must provide --layout or --ants")

    channels = _cli.parse_channels(channels, freq_range)
    
    # Build analytic_beam config if specified
    analytic_beam = None
    analytic_beam_map_file_path = None
    if analytic_beam_map_file is not None:
        analytic_beam_map_file_path = Path(analytic_beam_map_file)
        beam_map_csv = None  # multi-beam overrides UVBeam csv
        beamvar_type = None
        logger.info(f"Using multi-beam analytical mode from: {analytic_beam_map_file_path}")
    elif analytic_beam_class is not None:
        analytic_beam = build_analytic_beam_config(
                        beam_class=analytic_beam_class,
                        diameter=analytic_beam_diameter,
                        sigma=analytic_beam_sigma,
                        ref_freq=analytic_beam_ref_freq,
                        spectral_index=analytic_beam_spectral_index,
                        coeffs_file=analytic_beam_coeffs_file,
                        preset=analytic_beam_preset,
                        )
        # Analytical beams override per-antenna beams
        beam_map_csv = None
        beamvar_type = None
        logger.info(f"Using analytical beam: {analytic_beam_class}")

    # Gate per-antenna beams by simulator capability: matvis allows any number of
    # beams; fftvis allows a single-unique-beam CSV (one beam for the whole array)
    # and errors on genuinely per-antenna (multi-beam) CSVs.
    beam_map_csv = gate_beam_map_for_simulator(beam_map_csv, simulator, context="[runsim]")

    if "beam_interpolator" in kwargs:
        kwargs.pop("beam_interpolator")

    # sky_realization = kwargs.pop("sky_realization", None)

    run_validation_sim(
        layout=layout, 
        ants=ants, 
        channels=channels, 
        simulator=simulator,
        beam_map_csv=beam_map_csv,
        beamvar_type=beamvar_type,
        analytic_beam=analytic_beam,
        analytic_beam_map_file=analytic_beam_map_file_path,
        **kwargs)


@cli.command("make-obsparams")
@_cli.opts.layout
@_cli.opts.ants
@_cli.opts.ideal_layout
@_cli.opts.channels
@_cli.opts.freq_range
@_cli.opts.sky_model
@_cli.opts.sky_realization  
@_cli.opts.n_time_chunks
@_cli.opts.spline_interp_order
@_cli.opts.redundant
@_cli.opts.do_time_chunks
@click.option("--beam-interpolator", default='az_za_map_coordinates')
@click.option("--simulator", type=click.Choice(["fftvis", "matvis", "fftvis64", "fftvis32", "matvis-cpu"]), default="matvis", help="Pick the visibility backend; per-antenna beams are only admitted for matvis.")
@click.option("--beam-map-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="CSV with columns: ant_number,beam_file (per-antenna beams; matvis only).")
@click.option("--beamvar-type", type=click.Choice(['vivaldired', 'airyred', 'airyprb', 'airytilt']), help="Type of beam variation to use for non-redundant sims.")

# Analytical beam options start
@click.option("--analytic-beam-class", type=click.Choice([
    # "AiryBeam", "GaussianBeam", 
    "hera_sim.beams.PolyBeam", "hera_sim.beams.ZernikeBeam",
    "hera_sim.beams.PerturbedPolyBeam"
]), default=None, help="Analytical beam class to use")
@click.option("--analytic-beam-diameter", type=float, default=14.0, help="Diameter for AiryBeam (m)")
@click.option("--analytic-beam-sigma", type=float, default=0.15, help="Sigma for GaussianBeam (rad)")
@click.option("--analytic-beam-ref-freq", type=float, default=1.0e8, help="Reference freq (Hz)")
@click.option("--analytic-beam-spectral-index", type=float, default=-0.6975, help="Spectral index")
@click.option("--analytic-beam-coeffs-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--analytic-beam-preset", type=click.Choice(["fagnoni19", "custom"]), default=None)
# Analytical beam options end

def make_obsparams(
    layout, ants, ideal_layout, freq_range, channels, sky_model, sky_realization, n_time_chunks, 
    spline_interp_order, beam_interpolator, redundant, do_time_chunks,
    simulator, beam_map_csv, beamvar_type, 
    # analytical beam params
    analytic_beam_class, analytic_beam_diameter, analytic_beam_sigma,
    analytic_beam_ref_freq, analytic_beam_spectral_index,
    analytic_beam_coeffs_file, analytic_beam_preset,
):
    """Make obsparams for H4C simulations given a sky model and frequencies."""
    from core.obsparams import make_hera_obsparam
    
    # Resolve layout/ants precedence 
    if ants and layout:
        raise click.BadParameter("Do not provide both --layout and --ants")
    if ants:
        layout = ants
    if not ants and not layout:
        raise click.BadParameter("You must provide --layout or --ants")
    
    channels = _cli.parse_channels(channels, freq_range)
    
    # Build analytic_beam dict if class is specified
    analytic_beam = None
    if analytic_beam_class is not None:
        analytic_beam = build_analytic_beam_config(
            beam_class=analytic_beam_class,
            diameter=analytic_beam_diameter,
            sigma=analytic_beam_sigma,
            ref_freq=analytic_beam_ref_freq,
            spectral_index=analytic_beam_spectral_index,
            coeffs_file=analytic_beam_coeffs_file,
            preset=analytic_beam_preset,
        )
        # Analytical beams don't use beam_map_csv
        beam_map_csv = None
        logger.info(f"Using analytical beam: {analytic_beam_class}")
    
    # Gate per-antenna beams by simulator capability (give a helpful error early).
    beam_map_csv = gate_beam_map_for_simulator(beam_map_csv, simulator, context="[make-obsparams]")
         
    # ants-layout test
    print("do_time_chunks are ", do_time_chunks)
    print("layout passed is ", layout)
    # print("layout_user is ", layout_user)
    print("ideal_layout is ", ideal_layout)
    print("channels are ", channels)
    # print("ants-layout is ", ants)

    make_hera_obsparam(
        layout=layout,
        ideal_layout=ideal_layout,
        channels=channels,
        sky_model=sky_model,
        chunks=n_time_chunks,
        spline_interp_order=spline_interp_order,
        beam_interpolator=beam_interpolator,
        redundant=redundant,
        do_chunks=do_time_chunks,
#        simulator=simulator,
        beam_map_csv=beam_map_csv,
        beamvar_type=beamvar_type,
        analytic_beam=analytic_beam,
        sky_realization=sky_realization,
    )

# print("test 349678")
option_nside = click.option("--nside", default=256, show_default=True)


@cli.command("sky-model")
@click.argument("sky_model", type=click.Choice(["gsm", "diffuse", "ptsrc", "grf-eor"]))
@_cli.opts.channels
@_cli.opts.freq_range
@_cli.opts.slurm_override
@_cli.opts.skip_existing
@_cli.opts.dry_run
@option_nside
@click.option("--local/--slurm", default=False)
@click.option("--split-freqs/--no-split-freqs", default=False)
@click.option("--label", default="")
@click.option("--with-confusion/--no-confusion", default=True)
@click.option("--seed", type=int, default=2038, show_default=True,
              help="GRF realization seed (grf-eor only); selects which raw realization to read. "
                   "Must match the seed passed to `grf-realization`.")
@click.option("--realization", default=None,
              help="Name of the output realization subfolder (grf-eor only). Defaults to "
                   "'seed{seed}'. Override to reproduce richer names, e.g. "
                   "rlzn_seed_222_offsetfix_freqslic_grf.")
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
    label,
    with_confusion,
    seed,
    realization,
):
    """Make SkyModel at given frequencies.

    Frequencies are based on H4C data.
    Outputs are written to the default directories, i.e. "./sky_models/<type>".
    """

    channels = _cli.parse_channels(channels, freq_range)
    if local:
        from core import sky_model as sm

        if sky_model == "gsm":
            sm.make_gsm_model(channels, nside, label=label)
        elif sky_model == "diffuse":
            sm.make_diffuse_model(channels, nside, with_confusion=with_confusion, label=label)
        elif sky_model == "ptsrc":
            sm.make_ptsrc_model(channels, nside, label=label)
        elif sky_model == "grf-eor":
            sm.make_grf_eor_model(
                # f"healpix-maps{nside}{label}.h5",
                # Seed-tagged raw realization in sky_models/raw/ (read there directly; no copy).
                f"eor-grf-nside{nside}_seed{seed}.h5",
                channels=channels,
                label=label,
                seed=seed,
                realization=realization,
                # offset_mode="constant", offset_value=1e-5,
                offset_mode="shift_min", floor_epsilon=1e-6,
            )
        else:
            raise ValueError(f"Unknown sky model: {sky_model}")
    else:
        from core.run_sky_model import run_make_sky_model
        run_make_sky_model(
            sky_model,
            channels,
            nside,
            slurm_override=slurm_override,
            skip_existing=skip_existing,
            dry_run=dry_run,
            split_freqs=split_freqs,
            label=label,
            with_confusion=with_confusion,
            seed=seed,
            realization=realization,
        )
    
@cli.command
@click.option('--nside', type=int, required=True)
@click.option('--seed', type=int, default=2038)
@click.option("--low-memory/--fast-cpu", default=True)
@click.option("--local/--slurm", default=False)
def grf_realization(nside, seed, local, low_memory):
    from core.grf_realization import run_compute_grf_realization
    run_compute_grf_realization(nside=nside, seed=seed, low_memory=low_memory)

@cli.command
@click.option('--test-mode/--production', default=False)
@click.option('--ell-max', default=1250)
@click.option("--local/--slurm", default=False)
def grf_covariance(test_mode, ell_max, local):
    from core.grf_covariance import compute_grf_covariance, run_compute_grf_covariance
    
    if local:
        compute_grf_covariance(test_mode, ell_max=ell_max)
        print("running compute_grf_covariance in test mode ", test_mode)
    else:
        run_compute_grf_covariance(test_mode, ell_max=ell_max)
        print("running run_compute_grf_covariance in test mode", test_mode)
        print("freqs are ", len(1e-6*utils.FREQS_DICT['H6C']), 1e-6*utils.FREQS_DICT['H6C'])
        
@cli.command("cornerturn")
@_cli.opts.sky_model
@_cli.opts.sky_realization
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
@_cli.opts.redundant
@_cli.opts.prefix
def cornerturn(
    sky_model,
    sky_realization,
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
    redundant: bool,
    prefix: str    
):
    """Perform a cornerturn on simulation files.

    This takes multiple files, each with a single frequency and many times (snapshots),
    and reforms them into files with all frequencies and a set number of times (generally
    smaller). Note that the input files may be partial in frequency *and* time.

    Output files have the following prototype:
        zen.LST.{lst:.7f}[.{sky_cmp}].uvh5
    """
    logger.setLevel(log_level)

    sky_model_key = f"{sky_model}/{sky_realization}" if sky_realization else sky_model

    # Make sure that the slurm log directory exists.
    # Otherwise, the job will terminate
    log_dir = Path(f"logs/chunk/{sky_model_key}")
    log_dir.mkdir(parents=True, exist_ok=True)

    if direc is None:
        simdir = utils.OUTDIR / utils.get_direc(
            sky_model=sky_model_key, chunks=nchunks_sim, layout=layout,
            redundant=redundant, prefix=prefix,
        )
    else:
        simdir = Path(direc)

    outdir = simdir / "rechunk"
    outdir.mkdir(parents=True, exist_ok=True)

    conjugate = "--conjugate" if conjugate else ""
    remove_cross_pols = "--remove-cross-pols" if remove_cross_pols else ""

    if channels is None:
        print(simdir)
        print(simdir.glob("*"))
        allfiles = sorted(simdir.glob(f"fch????_chunk{time_chunk:05d}.uvh5"))
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
        ("mem", "31GB"),
        ("time", estimated_time),
    )

    sbatch = _cli._get_sbatch_program(gpu=False, slurm_override=slurm_override)

    cmd = f"""
    time python core/rechunk-fast.py \
    --r-prototype "fch{{channel:04d}}_chunk{time_chunk:05d}.uvh5" \
    --chunk-size {new_chunk_size} \
    --channels {channels} \
    --sky-cmp {sky_model_key}\
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

    sbatch_file = sbatch_dir / f"{sky_model_key}_ch{time_chunk:03d}_{layout}.sbatch"

    sbatch = "\n".join([sbatch, "", cmd, ""])
    with open(sbatch_file, "w") as fl:
        fl.write(sbatch)

    if not dry_run:
        subprocess.call(f"sbatch {sbatch_file}".split())

    logger.debug(f"\n===Job Script===\n{sbatch}\n===END===\n")


if __name__ == "__main__":
    cli()
