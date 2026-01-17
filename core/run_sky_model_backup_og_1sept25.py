from . import utils
from ._cli_utils import _get_sbatch_program
import logging
import subprocess

logger = logging.getLogger(__name__)

def run_make_sky_model(
    sky_model: str,
    channels: list[int],
    nside: int,
    slurm_override: tuple[tuple[str, str]],
    skip_existing: bool,
    dry_run: bool,
    split_freqs: bool = False,
    label: str = "",
    with_confusion: bool = True,
):
    """Run the sky model creation via SLURM."""
    model = f"{sky_model}{nside}"
    out_dir = utils.SKYDIR / f"{model}"
    logdir = utils.LOGDIR / f"skymodel/{model}"

    logdir.mkdir(parents=True, exist_ok=True)

    # We want to override the job-name to be <sky_model>-<fch>-<ch>, but the last two
    # variables have to be accessed in the loop, so we will be instead override it to
    # a Python string formatting pattern and format it in the loop.
    # Note that click default `slurm_overrride` to (), and we want it to be "2D" tuple
    slurm_override = slurm_override + (
        ("job-name", "{sky_model}-fch{fch:04d}" if split_freqs else sky_model),
        (
            "output",
            "{logdir}/fch{fch:04d}_%J.out" if split_freqs else "{logdir}/%J.out",
        ),
    )

    if "time" not in [x[0] for x in slurm_override]:
        slurm_override = slurm_override + (("time", "0-00:15:00"),)

    # Make the SBATCH script minus hera-sim-vis.py command
    program = _get_sbatch_program(gpu=False, slurm_override=slurm_override)

    sbatch_dir = utils.REPODIR / "batch_scripts/skymodel"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    if split_freqs:
        for fch in channels:
            logger.info(f"Working on frequency channel {fch}")

            outfile = out_dir / f"fch{fch:04d}.skyh5"

            # Check if output file already existed, if clobber is False
            if skip_existing and outfile.exists():
                logger.warning(f"File {outfile} exists, skipping")
                continue

            cmd = f"time python vsim.py sky-model {sky_model} --local --nside {nside} --freq-range {fch} {fch+1} --label '{label}'"

            if utils.HPC_CONFIG["slurm"]:
                # Write job script and submit
                sbatch_file = sbatch_dir / f"{sky_model}_fch{fch:04d}.sbatch"

                logger.info(f"Creating sbatch file: {sbatch_file}")
                # Now, join the job script with the hera-sim-vis.py command
                # and format the job-name
                job_script = "\n".join([program, "", cmd, ""]).format(
                    sky_model=sky_model, fch=fch, logdir=logdir
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
    else:
        channels = sorted(channels)
        groups = [[channels[0]]]
        for ch in channels[1:]:
            if ch == groups[-1][-1] + 1:
                groups[-1].append(ch)
            else:
                groups.append([ch])

        chan_opt = "".join(
            (
                f"--channels {g[0]} "
                if len(g) == 1
                else f"--channels {g[0]}~{g[-1] + 1}"
            )
            for g in groups
        )
        cmd = f"time python vsim.py sky-model {sky_model} --local --nside {nside} --label '{label}' {chan_opt}"

        if utils.HPC_CONFIG["slurm"]:
            # Write job script and submit
            sbatch_file = sbatch_dir / f"{sky_model}_allfreqs.sbatch"

            logger.info(f"Creating sbatch file: {sbatch_file}")
            job_script = "\n".join([program, "", cmd, ""]).format(
                sky_model=sky_model, logdir=logdir
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
