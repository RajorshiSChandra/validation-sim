#!/usr/bin/env python3
import os
from pathlib import Path
import logging
import numpy as np
import yaml
import click
import utils

H4C_FREQS = utils.h4c_freqs
REPODIR = utils.REPODIR

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def _get_sbatch(hpc_config: str | Path):
    with open(hpc_config, "r") as fl:
        hpc_params = yaml.safe_load(fl)
    conda_params = hpc_params["conda"]
    slurm_params = hpc_params["slurm"]
    # If H4c
    slurm_params["log_dir"] = "logs"
    slurm_params["log_name"] = "job_name"
    module = hpc_params["module"]

    sbatch_header = """#!/bin/bash
#SBATCH -o {log-dir}/{log-name}-%j.out
#SBATCH --job-name={job-name}
#SBATCH --partition={partition}
#SBATCH --mem={mem}GB
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus-per-task}
#SBATCH --time={time}
""".format_map(
        slurm_params
    )

    conda_setup = """source {conda_path}/bin/activate
conda activate {environment_name}
""".format_map(
        conda_params
    )

    module_setup = "".join([f"module load {md}\n" for md in module])

    sbatch = "\n".join([sbatch_header, conda_setup, module_setup])
    return sbatch


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
    type=click.IntRange(0, len(H4C_FREQS)),
    help="Frequency channel range to simulate",
)
@click.option(
    "-fch",
    "--freqs",
    multiple=True,
    default=None,
    type=click.IntRange(0, len(H4C_FREQS)),
    help="Frequency channel, allow multiple, add to --freq-range",
)
@click.option(
    "-sm",
    "--sky_model",
    default="ptsrc",
    show_default=True,
    type=click.Choice(["ptsrc", "diffuse_nside128", "eor"], case_sensitive=True),
    help="Sky model to simulate",
)
@click.option(
    "--chunks",
    default=3,
    show_default=True,
    help="Split simulation into a number of time chunks",
)
@click.option("--gpu/--cpu", default=False, show_default=True, help="Use gpu or cpu")
@click.option(
    "--clobber", is_flag=False, help="Do not check if outputs exist and overwrite"
)
# @click.option("--runtime", default=None, help="time string for slurm sbatch")
# @click.option("--mem", default=None, help="mem in GB for slurm sbatch")
# @click.option("--ntasks", default=None, help="ntasks for slurm sbatch")
# @click.option("--cpus-per-task", default=None, help="cpus per task for slurm sbatch")
@click.option(
    "--hpc-config",
    type=click.Path(exists=True, resolve_path=True),
    help="HPC config yaml file, run hera-sim-vis.py locally if not provided",
)
def h4c(freq_range, freqs, sky_model, chunks, gpu, clobber, hpc_config):
    """Run H4C validation simulations.

    Simulation parameters are fixed to
    * Vivaldi beam, extrapolated to cover 45-50 MHz
    * Beam interpolation: cubic in frequency, linear in spatail
    * Frequencies. See freqs.yaml
    """
    out_dir = REPODIR / f"outputs/{sky_model}/"
    config_dir = REPODIR / f"config_files/obsparams/{sky_model}/nt17280_spl{chunks}"
    simulator_config = REPODIR / "viscpu.yaml" if gpu else REPODIR / "viscpu.yaml"

    sbatch = _get_sbatch(hpc_config) if hpc_config else ""
    hera_sim_options = "--normalize_beams --fix_autos --log-level INFO"

    freq_chans = np.arange(*freq_range)
    if freqs:
        extra_freqs = freqs[np.isin(freqs, freq_chans, invert=True)]
        freq_chans = np.append(freq_chans, extra_freqs)
    for fch in freq_chans:
        for ch in range(chunks):
            # TODO: later remove nt17280
            outfile = out_dir / f"{sky_model}_fch{fch:04d}_nt17280_chunk{ch}.uvh5"
            # Check if output file already existed, if clobber is False
            if not clobber and outfile.exists():
                logging.warn(f"File {outfile} exists, skipping")
            else:
                # Check if obsparam exist
                # TODO: later remove nt1780
                obsparam = config_dir / f"fch{fch:04d}_chunk{ch}.yaml"
                if not obsparam.exists():
                    logging.warn(f"{obsparam} does not exist, skipping")
                    # TODO: later add option to automatically make obsparam
                else:
                    # Make sbatch script
                    cmd = f"hera-sim-vis.py {hera_sim_options} {obsparam} {simulator_config}"
                    if hpc_config:
                        job_script = "\n".join([sbatch, cmd])
                        print(job_script)
                        # Submit job
                        # with open("_sim.sbatch", "w") as fl:
                        # fl.write(job_script)

                        # os.system("sbatch _sim.sbatch")
                    else:
                        os.system(f"{cmd}")


@run_sim.command("custom")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["INFO", "WARNING", "ERROR", "DEBUG"], case_sensitive=True),
)
def custom(log_level):
    """Run custom simulations (not yet implemented).

    All configs to be provided.
    """
    pass


if __name__ == "__main__":
    run_sim()
