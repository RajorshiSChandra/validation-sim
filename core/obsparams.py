"""Functions for creating obsparam files."""

from functools import cache
from hashlib import md5
from pathlib import Path
import yaml
import numpy as np

from . import utils

H4C_FREQS = utils.FREQS_DICT["H4C"]
CFGDIR, SKYDIR, OUTDIR = utils.CFGDIR, utils.SKYDIR, utils.OUTDIR
NTIMES, INTEGRATION, START_TIME = (
    utils.VALIDATION_SIM_NTIMES,
    utils.VALIDATION_SIM_INTEGRATION_TIME,
    utils.VALIDATION_SIM_START_TIME,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@cache
def make_tele_config(
    freq_interp_kind: str = "cubic", spline_interp_order: int = 3, beam_interpolator: str = "az_za_map_coordinates"
) -> Path:
    """Make a telescope config file."""
    config = f"""
beam_paths:
  0: '{utils.BEAMDIR}/NF_HERA_Vivaldi_efield_beam_extrap.fits'
telescope_location: {str(utils.HERA_LOC)}
telescope_name: HERA
freq_interp_kind: '{freq_interp_kind}'
"""

    if beam_interpolator=="az_za_simple":
        config += f"""
spline_interp_opts:
  kx: {spline_interp_order}
  ky: {spline_interp_order}
"""
    elif beam_interpolator=="az_za_map_coordinates":
        config += f"""
spline_interp_opts:
  order: {spline_interp_order}
"""


    _fname = f"hera_{freq_interp_kind}_{spline_interp_order}.yaml"
    fname = CFGDIR / "teleconfigs" / "tmp" / _fname

    fname.parent.mkdir(exist_ok=True, parents=True)
    with open(fname, "w") as fl:
        fl.write(config)

    return fname


def quoted_presenter(dumper, data):
    """Represent a string in quotes."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")


yaml.add_representer(str, quoted_presenter)


def make_hera_obsparam(
    layout: str | list[int] | Path,
    channels: list[int],
    sky_model: str,
    chunks: int,
    do_chunks: list[int] | None = None,
    ideal_layout: bool = True,
    freq_interp_kind: str = "cubic",
    spline_interp_order: int = 3,
    beam_interpolator: str = "az_za_map_coordinates",
    season: str = "H4C",
    force: bool = False,
    redundant: bool = False,
    prefix: str = "default"
):
    """Create an obsparam file."""
    freq_vals = utils.FREQS_DICT[season][channels]

    if NTIMES % chunks != 0:
        raise ValueError(f"Please choose chunks to divide NTIMES {NTIMES} cleanly")

    print('chunks: ', chunks)
    if do_chunks is None:
        do_chunks = list(range(chunks+1))
    else:
        assert all(x < chunks for x in do_chunks)
    print(do_chunks)
    Ntimes_per_chunk = NTIMES // chunks

    if isinstance(layout, str):
        # it's a name
        layout_file = utils.make_hera_layout(name=layout, ideal=ideal_layout)
    elif isinstance(layout, Path):
        layout_file = layout
    else:
        # it's a list of integers specifying antennas
        layout_file = utils.make_hera_layout(
            name=f"HERA_custom_subset_{md5(str(layout).encode()).hexdigest()}",
            ants=layout,
            ideal=ideal_layout,
        )

    tele_config_file = make_tele_config(
        freq_interp_kind=freq_interp_kind, spline_interp_order=spline_interp_order, beam_interpolator=beam_interpolator,
    )

    modeldir = utils.get_direc(
        sky_model=sky_model, chunks=chunks, layout=layout_file.stem,
        redundant=redundant, prefix=prefix,
    )
    
    obsparams_dir = utils.OBSPDIR / modeldir
    obsparams_dir.mkdir(parents=True, exist_ok=True)
    outdir = utils.OUTDIR / modeldir
    outdir.mkdir(parents=True, exist_ok=True)

    if redundant:
        redfile = layout_file.with_suffix(".redundancies")
        if redfile.exists():
            redbls = np.genfromtxt(redfile)
            
        else:
            from pyuvdata.utils.redundancy import get_antenna_redundancies
            from pyuvdata.utils import baseline_to_antnums

            ants = np.genfromtxt(layout_file, skip_header=1, usecols=(1, 3,4,5), delimiter='\t')
            antnums = ants[:, 0]
            redbls = get_antenna_redundancies(antnums, ants[:, 1:], tol=4.0, use_grid_alg=True, include_autos=True)[0]  # hera thresh 
            redbls = np.array([baseline_to_antnums(r[0], Nants_telescope=350) for r in redbls])
            np.savetxt(redfile, redbls)
        reds = [(int(a), int(b)) for a, b in redbls]

    print(channels, freq_vals, do_chunks)
    for fch, fv in zip(channels, freq_vals):
        for ch in do_chunks:
            jobname = modeldir / utils.get_file(chunk=ch, channel=fch, with_dir=False)
            obsparams_file = utils.OBSPDIR / jobname
            print(f"Going to make {obsparams_file}")
            if obsparams_file.exists() and not force:
                continue

            # Note that global paths from utils are Path objects. f-string formatting
            # automatically converts them to string for yaml to write out.
            obsparams = {
                "filing": {
                    "outdir": f"{outdir}",
                    "outfile_name": jobname.name,
                    "output_format": "uvh5",
                    "clobber": True,
                },
                "freq": {
                    "Nfreqs": 1,
                    "channel_width": float(
                        utils.FREQS_DICT[season][1] - utils.FREQS_DICT[season][0]
                    ),
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
                'cat_name': sky_model,
            }
            
            if redundant:
                obsparams['select'] = {'bls': str(reds)}

            with open(obsparams_file, "w") as stream:
                yaml.dump(obsparams, stream, default_flow_style=False, sort_keys=False)

            print(f"Wrote obsparams at {obsparams_file}")

    return layout_file
