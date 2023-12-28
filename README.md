# HERA Validation Simulation Scripts

High-level scripts for performing HERA Validation (ideal) simulations, as well as 
utilities for performing similar custom simulations (for performance benchmarking,
debugging, etc.)

## How to use

### Initial Setup

1. Clone this repo, and create a Python environment using the included environment file.

    ```conda env create -f environment.yaml```

2. If not already present somewhere on your system, clone the `HERA-Beams` repo to
   a location *outside* this repo:

    ```git clone git@github:hera-team/HERA-Beams ../hera-beams```

3. If your system has no HPC configuration file in `hpc-configs/` yet, add one
   (cp/paste one of the existing ones). Specifically:

    * Change the `paths.beams` entry to point to the beams repo location 
      (adding `NicolasFagnoniBeams` to the end of the string).
    * Add any modules required to be loaded on your system (if any). You'll need a
      version of `cuda` and a version of `openmpi`.
    * Update both the `cpu` and `gpu` entries of the `slurm:` section. These provide
      _defaults_ for running jobs (mostly simulations) on the cluster via SLURM.
      The most important entries are the `partition`, `nodes`, `mem` and `gres` (or
      `gpus`). In general, you should use as much `mem` as available on a single node
      on your system, and use `nodes=1` (as multiple nodes are used via multiple
      jobs).

### Running a Simulation
Several general steps need to be taken to run a particular simulation. These can all
be run at once by using `run_sim.py`, but sometimes it is preferable to do it one step at a time.

1. Make sky models by running `python make_sky_model.py [SKYMODEL]`. You can use the
   `--help` menu to find more info. This (currently) creates the sky models directly
   on whatever your login node is. In the future, this might be updated to pass jobs
   to SLURM. One file per frequency channel is output into the `sky_models/` directory.

2. Make simulation configuration (obsparams) with `python make_obsparams.py`. 
   There are several options here, so be sure to use the `--help` menu. Also, the
   obsparams are created automatically if you run the next step, so there is no real
   need to run this step individually. Some of the important options are `--layout`,
   which is a string name (check the `help` menu to see what is available currently)
   that maps onto a particular antenna layout (usually a specific subset of HERA 350). 
   You can add your own by updating the `utils.py` module. You can *instead*
   provide `-a [ANTNUM]` any number of times to include only those antennas from the
   full HERA 350. You can also provide `--freq-range` and/or `--freqs` to specify
   which channels to include (all frequency channels are from the actual HERA array).
   Also provide the `--sky-model` in the same way as step 1. Finally, provide `--chunks` 
   to set the number of chunks over which the 17280 LSTs will be simulated. Using a 
   higher number can be good for debugging on small chunks.

3. Run the simulation with `python run_sim.py`. Again, `--help` is your friend. Here,
   you can (in addition to all options from `make_obsparams.py` above) specify 
   `--do-time-chunks` which limits the actual time chunks to simulate (i.e. you might 
   split the simulation into 288 chunks, and only perform the first one). You can also 
   specify `--dry-run` to create all the configuration files and sbatch files, but not 
   actually run the simulation.

### Other Utilities

* `rechunk-fast.py`: This performs cornerturns on the output of the `run_sim.py` 
  command, taking in the files with large number of times and a single frequency and 
  outputing files with smaller number of times and all frequencies, ready to be 
  passed on to systematics simulation. Run this script using `run_rechunk.sh`.
* `notebooks/`: a bunch of notebooks used for validating the outputs.