#!/usr/bin/env python3
import subprocess
from pathlib import Path
import logging
import numpy as np
import yaml
import click
import utils
from make_obsparams import make_h4c_obsparam


H4C_FREQS = utils.H4C_FREQS
REPODIR = utils.REPODIR

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=100)

logger = logging.getLogger(__name__)


def _get_sbatch_program(hpc_config: str | Path, slurm_override=None):
    with open(hpc_config, "r") as fl:
        hpc_params = yaml.safe_load(fl)
    conda_params = hpc_params["conda"]
    module_params = hpc_params["module"]
    slurm_params = hpc_params["slurm"]
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


@click.group(context_settings=CONTEXT_SETTINGS)
def run_sim():
    """For grouping click commands"""
    pass


@run_sim.command("h4c")
@click.option(
    "-fr",
    "--freq-range",
    nargs=2,
    default=(0, len(H4C_FREQS)),
    show_default=True,
    metavar="START STOP",
    type=click.IntRange(0, len(H4C_FREQS)),
    help="Frequency channel range (Pythonic)",
)
@click.option(
    "-fch",
    "--freqs",
    multiple=True,
    default=None,
    type=click.IntRange(0, len(H4C_FREQS)),
    metavar="FREQ_CHAN",
    help="Frequency channel, allow multiple, add to --freq-range",
)
@click.option(
    "-sm",
    "--sky_model",
    default="ptsrc",
    show_default=True,
    type=click.Choice(["ptsrc", "diffuse_nside256", "eor"], case_sensitive=True),
    help="Sky model to simulate",
)
@click.option(
    "--chunks",
    default=3,
    show_default=True,
    help="Split the simulation into a number of time chunks",
)
@click.option("--gpu/--cpu", default=False, show_default=True, help="Use gpu or cpu")
@click.option(
    "--hpc-config",
    type=click.Path(exists=True, resolve_path=True),
    help="HPC config YAML file for creating an SBATCH script "
    "to run on a cluster; run locally if not provided",
)
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
    "--make-missing-obsparam/--skip-missing-obsparam",
    default=True,
    show_default=True,
    help="Make the obsparam or skip running the simulation if the obsparam is missing",
)
@click.option(
    "--remake-all-obsparams",
    is_flag=True,
    help="If set, remake all obsparams before running the simulations",
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["INFO", "WARNING", "ERROR", "DEBUG"], case_sensitive=True),
    show_default=True,
    help="Verbosity level, also pass the flag to hera-sim-vis.py",
)
@click.option("-d", "--dry-run", is_flag=True, help="Pass the flag to hera-sim-vis.py")
def h4c(
    freq_range,
    freqs,
    sky_model,
    chunks,
    gpu,
    hpc_config,
    slurm_override,
    skip_existing,
    make_missing_obsparam,
    remake_all_obsparams,
    log_level,
    dry_run,
):
    """Make a job script to run H4C validation sims via hera-sim-vis.py.

    Use the default parameters, configuration files, and directories for H4C sims
    (see make_obsparams.py).
    """
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, log_level)
    )

    out_dir = REPODIR / f"outputs/{sky_model}/nt17280/"
    config_dir = REPODIR / f"config_files/obsparams/{sky_model}/nt17280_spl{chunks}"
    simulator_config = REPODIR / "visgpu.yaml" if gpu else REPODIR / "viscpu.yaml"

    # We want to override the job-name to be <sky_model>-<fch>-<ch>, but the last two
    # varaibles have to be accessed in the loop, so we will be instead override it to
    # a Python string formatting pattern and format it in the loop.
    # Note that click default `slurm_overrride` to (), and we want it to be "2D" tuple
    slurm_override = slurm_override + (
        ("job-name", "{sky_model}-fch{fch:04d}-chunk{ch}"),
    )
    # Make the SBATCH script minus hera-sim-vis.py command
    program = _get_sbatch_program(hpc_config, slurm_override) if hpc_config else ""

    # Option for hera-sim-vis.py. Let's just keep this fixed.
    sim_options = f"--normalize_beams --fix_autos --log-level {log_level}"
    sim_options += " --dry-run" if dry_run else ""

    freq_chans = np.arange(*freq_range)
    if freqs:
        extra_freqs = freqs[np.isin(freqs, freq_chans, invert=True)]
        freq_chans = np.append(freq_chans, extra_freqs)

    if remake_all_obsparams:
        make_h4c_obsparam(freq_range, freqs, sky_model, chunks)

    for fch in freq_chans:
        for ch in range(chunks):
            # TODO: later remove nt17280
            outfile = out_dir / f"{sky_model}_fch{fch:04d}_nt17280_chunk{ch}.uvh5"
            # Check if output file already existed, if clobber is False
            if skip_existing and outfile.exists():
                logger.warning(f"File {outfile} exists, skipping")
            else:
                # TODO: later remove nt1780
                obsparam = config_dir / f"fch{fch:04d}_chunk{ch}.yaml"
                if not obsparam.exists():
                    if make_missing_obsparam:
                        make_h4c_obsparam(
                            freq_range=(fch, fch + 1),
                            freqs=None,
                            sky_model=sky_model,
                            chunks=chunks,
                        )
                    else:
                        logger.warning(f"{obsparam} does not exist, skipping")
                else:
                    cmd = f"hera-sim-vis.py {sim_options} {obsparam} {simulator_config}"
                    if hpc_config:
                        # Now, join the job script with the hera-sim-vis.py command
                        # and format the job-name
                        job_script = "\n".join([program, "", cmd, ""]).format(
                            sky_model=sky_model, fch=fch, ch=ch
                        )
                        # Write job script and submit
                        # Maybe we should save each job script individually
                        with open("_sim.sbatch", "w") as fl:
                            fl.write(job_script)

                        subprocess.call(
                            "sbatch _sim.sbatch".split()
                        ) if not dry_run else None
                        logger.info(f"\n===Job Script===\n{job_script}\n===END===\n")
                    else:
                        logger.info("Running the simulation locally\nCommand: {cmd}")
                        subprocess.call(cmd.split()) if not dry_run else None


@run_sim.command("custom")
def custom(log_level):
    """Run custom simulations (not yet implemented).

    All configs to be provided.
    """
    pass


if __name__ == "__main__":
    run_sim()
