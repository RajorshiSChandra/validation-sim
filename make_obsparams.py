import click
import yaml
import numpy as np

import utils

H4C_FREQS = utils.H4C_FREQS
H4C_CHANNEL_WIDTH = utils.H4C_CHANNEL_WIDTH
CFGDIR, SKYDIR, OUTDIR = utils.CFGDIR, utils.SKYDIR, utils.OUTDIR
H4C_TELE_CONFIG, H4C_ARRAY_LAYOUT = utils.H4C_TELE_CONFIG, utils.H4C_ARRAY_LAYOUT
NTIMES, INTEGRATION, START_TIME = (
    utils.VALIDATION_SIM_NTIMES,
    utils.VALIDATION_SIM_INTEGRATION_TIME,
    utils.VALIDATION_SIM_START_TIME,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def quoted_presenter(dumper, data):
    """Custom yaml representer for string"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")


yaml.add_representer(str, quoted_presenter)


@click.group(context_settings=CONTEXT_SETTINGS)
def make_obsparams():
    """For grouping click commands"""
    pass


# TODO: These options are the same as options in `run_sim.py`. We can define them as
# a variable in utils and do @variable.
@make_obsparams.command("h4c")
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
    type=click.Choice(["ptsrc", "diffuse_nside256", "eor"], case_sensitive=True),
    help="Sky model to simulate",
)
@click.option(
    "--chunks",
    default=3,
    show_default=True,
    help="Split simulation into a number of time chunks",
)
def h4c(freq_range, freqs, sky_model, chunks):
    """Make an obsparams for H4C sim given a sky model and frequencies."""
    _make_h4c_obsparam(freq_range, freqs, sky_model, chunks)


def _make_h4c_obsparam(freq_range, freqs, sky_model, chunks):
    """Logic of the h4c cli function.

    This allow the function to be called from other modules."""
    freq_chans = np.arange(*freq_range)
    if freqs:
        extra_freqs = freqs[np.isin(freqs, freq_chans, invert=True)]
        freq_chans = np.append(freq_chans, extra_freqs)
    freq_vals = H4C_FREQS[freq_chans]

    Ntimes_per_chunk = NTIMES // chunks

    obsparams_dir = CFGDIR / f"obsparams/{sky_model}/nt17280_spl{chunks}"
    obsparams_dir.mkdir(parents=True, exist_ok=True)

    for fch, fv in zip(freq_chans, freq_vals):
        for ch in range(chunks):
            # Note that global paths from utils are Path object. f-string formatting
            # automatically converty them to string for yaml to write out.
            obsparams = {
                "filing": {
                    "outdir": f"{OUTDIR}/{sky_model}/nt17280",
                    "outfile_name": f"{sky_model}_fch{fch:04d}_nt17280_chunk{ch}",
                    "output_format": "uvh5",
                    "clobber": True,
                },
                "freq": {
                    "Nfreqs": 1,
                    "channel_width": H4C_CHANNEL_WIDTH,
                    "start_freq": float(fv),
                },
                "sources": {"catalog": f"{SKYDIR}/{sky_model}/fch{fch:04d}.skyh5"},
                "telescope": {
                    "array_layout": f"{H4C_ARRAY_LAYOUT}",
                    "telescope_config_name": f"{H4C_TELE_CONFIG}",
                    "select": {"freq_buffer": 3.0e6},
                },
                "time": {
                    "Ntimes": Ntimes_per_chunk,
                    "integration_time": INTEGRATION,
                    "start_time": START_TIME
                    + INTEGRATION * ch * Ntimes_per_chunk / 86400,
                },
                # This order makes it fastest to put the vis-cpu data back in.
                "polarization_array": [-5, -7, -8, -6],
            }

            obsparams_file = f"{obsparams_dir}/fch{fch:04d}_chunk{ch}.yaml"

            with open(obsparams_file, "w") as stream:
                yaml.dump(obsparams, stream, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    make_obsparams()
