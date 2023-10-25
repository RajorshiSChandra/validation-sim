import click
import yaml
import numpy as np
from typing import Sequence
import utils
from hashlib import md5
from pathlib import Path

H4C_FREQS = utils.FREQS_DICT["H4C"]
CFGDIR, SKYDIR, OUTDIR = utils.CFGDIR, utils.SKYDIR, utils.OUTDIR
#H4C_TELE_CONFIG, H4C_ARRAY_LAYOUT = utils.H4C_TELE_CONFIG, utils.H4C_ARRAY_LAYOUT
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


def _make_hera_obsparam(
    layout: str | list[int] | Path, 
    freq_range: tuple[int, int], 
    freqs: Sequence[int], 
    sky_model: str, 
    chunks: int, 
    do_chunks: list[int] | None,
    freq_interp_kind: str = 'cubic', 
    spline_interp_order: int = 3,
    season: str = 'H4C',
    force: bool = False,
):
    """Logic of the h4c cli function.

    This allow the function to be called from other modules."""
    freq_chans = np.arange(*freq_range)
    if freqs:
        extra_freqs = freqs[np.isin(freqs, freq_chans, invert=True)]
        freq_chans = np.append(freq_chans, extra_freqs)
    freq_vals = utils.FREQS_DICT[season][freq_chans]

    if not NTIMES % chunks == 0:
        raise ValueError(f"Please choose chunks to divide NTIMES {NTIMES} cleanly")

    if do_chunks is None:
        do_chunks = list(range(chunks))
    else:
        assert all(x < chunks for x in do_chunks)

    Ntimes_per_chunk = NTIMES // chunks

    obsparams_dir = utils.OBSPDIR / utils.OBSPARAM_DIRFMT.format(
        sky_model=sky_model, chunks=chunks
    )
    obsparams_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(layout, str):
        # it's a name
        layout_file = utils.make_hera_layout(name=layout)
    elif isinstance(layout, Path):
        layout_file = layout
    else:
        # it's a list of integers specifying antennas
        layout_file = utils.make_hera_layout(
            name=f"HERA_custom_subset_{md5(str(layout).encode()).hexdigest()}", 
            ants=layout
        )

    tele_config_file = utils.make_tele_config(
        freq_interp_kind=freq_interp_kind, 
        spline_interp_order=spline_interp_order
    )

    outdir = utils.OUTDIR / utils.VIS_DIRFMT.format(sky_model=sky_model, chunks=chunks)

    for fch, fv in zip(freq_chans, freq_vals):
        for ch in do_chunks:
            obsparams_file = obsparams_dir / utils.OBSPARAM_FLFMT.format(fch=fch, ch=ch)

            if obsparams_file.exists() and not force:
                continue

            # Note that global paths from utils are Path objects. f-string formatting
            # automatically converts them to string for yaml to write out.
            obsparams = {
                "filing": {
                    "outdir": f"{outdir}",
                    "outfile_name": utils.VIS_FLFMT.format(sky_model=sky_model, fch=fch, ch=ch),
                    "output_format": "uvh5",
                    "clobber": True,
                },
                "freq": {
                    "Nfreqs": 1,
                    "channel_width": float(utils.FREQS_DICT[season][1] - utils.FREQS_DICT[season][0]),
                    "start_freq": float(fv),
                },
                "sources": {"catalog": f"{SKYDIR}/{sky_model}/fch{fch:04d}.skyh5"},
                "telescope": {
                    "array_layout": f"{layout_file}",
                    "telescope_config_name": f"{tele_config_file}",
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


            with open(obsparams_file, "w") as stream:
                yaml.dump(obsparams, stream, default_flow_style=False, sort_keys=False)

    return layout_file

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Make the obsparam configuration files for visibiliy simulations."""
    pass


# TODO: These options are the same as options in `run_sim.py`. We can define them as
# a variable in utils and do @variable.
@cli.command("h4c")
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
def h4c_cli(layout, ants, freq_range, freqs, sky_model, chunks):
    """Make obsparams for H4C simulations given a sky model and frequencies."""
    if layout is not None and ants is not None:
        raise ValueError("Provide only layout or ants")
    if layout is None and ants is None:
        raise ValueError("Either layout or ants must be provided")
    if ants is not None:
        layout = list(ants)

    _make_hera_obsparam(
        layout=layout, 
        freq_range=freq_range, 
        freqs=freqs, 
        sky_model=sky_model, 
        chunks=chunks
    )


if __name__ == "__main__":
    cli()
