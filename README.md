# H4C Validation Simulation Scripts

High-level scripts for simulation H4C validation simulations. Also, support custom simulations.

## How to use

1. Clone, and create a Python environment using the included environment file.

    ```conda env create -f environment.yaml```

2. Download the `gleam_like_fainter.npz` from lustre (and the HERA beam file if using)

3. Make sky models by using `make_sky_model.py`

4. Make simulation configuration (obsparams) with `make_obsparams.py`. Some telescope and array layout configuration files are provided in `config_files` directory. Manually add more.

5. Run the simulation with `run_sim.py`. If running on an HPC cluster, an HPC configuration file (see `hpc_config.yam`) can be customized and provided to the `run_sim.py` script.