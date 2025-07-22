"""Utilities for writing CLI's that process with SLURM."""
from . import utils
import logging
import subprocess
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

def _get_sbatch_program(gpu: bool, slurm_override: dict[str, Any] | None=None):
    """Format an SBATCH program from parts."""
    conda_params = utils.HPC_CONFIG["conda"]
    module_params = utils.HPC_CONFIG["module"]
    slurm_params = utils.HPC_CONFIG["slurm"]["gpu" if gpu else "cpu"]

    if slurm_override is not None:  # Modify slurm options
        for k, v in slurm_override.items():
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

    return "\n".join([shebang, sbatch, conda, module])

def slurmify(
    name: str | None = None,
    logdir: str | None = None,
    jobname: str | None = None,
    outname: str = "%J.out",
    time: str | None = None,
    defaultgpu: bool = False,
    defaultmem: str | None = None,
    defaulttasks: int | None = None,
    partition: str | None = None,
):
    def inner(fnc):
        
        @wraps(fnc)
        def wrapper(
            *args, 
            slurm_override: dict | None=None, 
            gpu: bool = defaultgpu, 
            dry_run: bool = False,
            
            **kwargs
        ):
            cmd = fnc(*args, **kwargs)

            if not utils.HPC_CONFIG['slurm']:
                logger.info(f"Running the simulation locally\nCommand: {cmd}")
                if not dry_run:
                    subprocess.call(cmd.split())
                return
            
            cmdname = name or fnc.__name__
            logname = logdir  or cmdname
            jobtitle = jobname or cmdname
                
            logname = utils.LOGDIR / logname
            logname.mkdir(parents=True, exist_ok=True)

            slurm_defaults = {
                'job-name': jobtitle,
                'output': f"{logname}/{outname}"
            }
            if time is not None:
                slurm_defaults['time'] = time
            if defaultmem is not None:
                slurm_defaults['mem'] = defaultmem
            if defaulttasks is not None:
                slurm_defaults['ntasks'] = defaulttasks
            if partition is not None:
                slurm_defaults['partition'] = partition
                
            if slurm_override is None:
                slurm_override = {}
                
            slurm_defaults |= slurm_override
            
            # Make the SBATCH script minus hera-sim-vis.py command
            program = _get_sbatch_program(gpu=gpu, slurm_override=slurm_defaults)

            sbatch_dir = utils.REPODIR / "batch_scripts" / cmdname
            sbatch_dir.mkdir(parents=True, exist_ok=True)
            
             
            # Write job script and submit
            sbatch_file = sbatch_dir / "job.sbatch"

            logger.info(f"Creating sbatch file: {sbatch_file}")
            
            job_script = "\n".join([program, "", cmd, ""])
            with open(sbatch_file, "w") as fl:
                fl.write(job_script)

            if not dry_run:
                subprocess.call(f"sbatch {sbatch_file}".split())

            logger.debug(f"\n===Job Script===\n{job_script}\n===END===\n")

        return wrapper
    return inner
        
        