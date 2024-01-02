import click
import yaml
import numpy as np
from typing import Sequence
from . import utils
from hashlib import md5
from pathlib import Path
from functools import cache

H4C_FREQS = utils.FREQS_DICT["H4C"]
CFGDIR, SKYDIR, OUTDIR = utils.CFGDIR, utils.SKYDIR, utils.OUTDIR
NTIMES, INTEGRATION, START_TIME = (
    utils.VALIDATION_SIM_NTIMES,
    utils.VALIDATION_SIM_INTEGRATION_TIME,
    utils.VALIDATION_SIM_START_TIME,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@cache
def make_tele_config(freq_interp_kind: str = 'cubic', spline_interp_order: int = 3) -> Path:
    beampath = utils.HPC_CONFIG['paths']['beams']
    config = f"""
beam_paths:
  0: '{beampath}/NF_HERA_Vivaldi_efield_beam_extrap.fits'
telescope_location: {str(utils.HERA_LOC)}
telescope_name: HERA
freq_interp_kind: '{freq_interp_kind}'
spline_interp_opts:
        kx: {spline_interp_order}
        ky: {spline_interp_order}
"""
    
    fname = CFGDIR / 'teleconfigs' / 'tmp' / f'hera_{freq_interp_kind}_{spline_interp_order}.yaml'

    fname.parent.mkdir(exist_ok=True, parents=True)
    with open(fname, 'w') as fl:
        fl.write(config)

    return fname

def quoted_presenter(dumper, data):
    """Custom yaml representer for string"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")


yaml.add_representer(str, quoted_presenter)


def make_hera_obsparam(
    layout: str | list[int] | Path, 
    channels: list[int], 
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
    freq_vals = utils.FREQS_DICT[season][channels]

    if NTIMES % chunks != 0:
        raise ValueError(f"Please choose chunks to divide NTIMES {NTIMES} cleanly")

    if do_chunks is None:
        do_chunks = list(range(chunks))
    else:
        assert all(x < chunks for x in do_chunks)

    Ntimes_per_chunk = NTIMES // chunks

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

    tele_config_file = make_tele_config(
        freq_interp_kind=freq_interp_kind, 
        spline_interp_order=spline_interp_order
    )

    obsparams_dir = utils.OBSPDIR / utils.OBSPARAM_DIRFMT.format(
        sky_model=sky_model, chunks=chunks, layout=layout_file.stem
    )
    obsparams_dir.mkdir(parents=True, exist_ok=True)

    outdir = utils.OUTDIR / utils.VIS_DIRFMT.format(
        sky_model=sky_model, chunks=chunks, layout=layout_file.stem
    )

    for fch, fv in zip(channels, freq_vals):
        for ch in do_chunks:
            obsparams_file = obsparams_dir / utils.OBSPARAM_FLFMT.format(
                fch=fch, ch=ch, layout=layout_file.stem, sky_model=sky_model
            )

            if obsparams_file.exists() and not force:
                continue

            # Note that global paths from utils are Path objects. f-string formatting
            # automatically converts them to string for yaml to write out.
            obsparams = {
                "filing": {
                    "outdir": f"{outdir}",
                    "outfile_name": utils.VIS_FLFMT.format(sky_model=sky_model, fch=fch, ch=ch, layout=layout_file.stem),
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
