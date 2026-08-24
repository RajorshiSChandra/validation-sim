"""Functions for creating obsparam files."""

import logging
from functools import cache
from hashlib import md5
from pathlib import Path
import yaml
import numpy as np
import re
import hashlib

from . import utils

logger = logging.getLogger(__name__)

H4C_FREQS = utils.FREQS_DICT["H4C"]
CFGDIR, SKYDIR, OUTDIR = utils.CFGDIR, utils.SKYDIR, utils.OUTDIR
NTIMES, INTEGRATION, START_TIME = (
    utils.VALIDATION_SIM_NTIMES,
    utils.VALIDATION_SIM_INTEGRATION_TIME,
    utils.VALIDATION_SIM_START_TIME,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# Load CSV file with BeamID per antenna
def read_beam_map_csv(beam_map_csv: Path) -> dict[int, Path]:
    """Return {ant_number: Path(beam_file)} from a 2-column CSV."""
    import csv
    mapping: dict[int, Path] = {}
    with open(beam_map_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "ant_number" not in reader.fieldnames or "beam_file" not in reader.fieldnames:
            raise ValueError("beam_map_csv must have column names: ant_number, beam_file")
        for row in reader:
            ant = int(row["ant_number"])
            mapping[ant] = Path(row["beam_file"]).expanduser().resolve()
    if not mapping:
        raise ValueError("beam_map_csv was empty.")
    return mapping


def count_unique_beam_files(beam_map_csv: Path) -> int:
    """Number of distinct (resolved) beam files referenced by a beam-map CSV."""
    return len(set(read_beam_map_csv(beam_map_csv).values()))


def gate_beam_map_for_simulator(beam_map_csv, simulator, *, context: str = ""):
    """Gate per-antenna beam maps by simulator capability.

    - matvis : any number of beams (true per-antenna).
    - fftvis : hera_sim FFTVis supports only ONE beam for the whole array
               (FFTVis.validate raises 'FFTVis only supports a single beam.').
               A CSV that resolves to exactly one unique file is allowed
               (single-beam array); >1 distinct beams raises a clear error
               rather than silently falling back to the default beam.

    Returns the (unchanged) beam_map_csv, or raises ValueError.
    """
    if beam_map_csv is None:
        return None
    is_matvis = isinstance(simulator, str) and simulator.lower().startswith("matvis")
    if is_matvis:
        return beam_map_csv
    n = count_unique_beam_files(beam_map_csv)
    if n == 1:
        logger.info(
            "%s --beam-map-csv resolves to a single unique beam; "
            "accepted for simulator=%r (single-beam array).",
            context, simulator,
        )
        return beam_map_csv
    raise ValueError(
        f"{context} --beam-map-csv maps antennas to {n} distinct beam files, but "
        f"simulator={simulator!r} supports only a single beam for the whole array. "
        f"Use --simulator matvis for per-antenna beams, or supply a single-beam CSV "
        f"(or --analytic-beam-*)."
    )


#def write_array_with_beamids(src_layout: Path, beamid_by_ant: dict[int, int]) -> Path:
    #"""
    #Read the tab-delimited layout produced by utils.make_hera_layout and
    #write a copy that includes a BeamID column. Return the new file path.
    #"""
    ## Expect TSV with header (delimiter='\t')
    #header = Path(src_layout).read_text().splitlines()[0].split('\t')
    ## Check for required columns
    #musthave = {"Name", "Number", "E", "N", "U"}
##    musthave = {'Name    Number  BeamID  E       N       U'}
    #if not musthave.issubset(set(header)):
        #raise ValueError(f"{src_layout} missing required columns {musthave}. Got: {header}")
#
    #lines = Path(src_layout).read_text().splitlines()
    #cols = header
    #need_inject = "BeamID" not in cols
    #if need_inject:
        ## insert BeamID after 'Number' for readability
        #insert_at = cols.index("Number") + 1
        #cols = cols[:insert_at] + ["BeamID"] + cols[insert_at:]
#
    #out = []
    #out.append('\t'.join(cols))
    ## Map name->index for existing columns
    #idx = {c: i for i, c in enumerate(lines[0].split('\t'))}
#
    #for line in lines[1:]:
        #if not line.strip():
            #continue
        #parts = line.split('\t')
        #ant_num = int(float(parts[idx["Number"]]))  # Number column may be float-y
        #beamid = beamid_by_ant.get(ant_num)
        #if beamid is None:
            #raise ValueError(f"No BeamID mapping provided for antenna Number={ant_num}")
        #if need_inject:
            #parts = parts[:idx["Number"]+1] + [str(beamid)] + parts[idx["Number"]+1:]
        #else:
            #parts[idx["BeamID"]] = str(beamid)
        #out.append('\t'.join(parts))
#
    #dst = src_layout.with_suffix(src_layout.suffix + ".with_beamids")
    #Path(dst).write_text('\n'.join(out) + '\n')
    #return dst

def write_array_with_beamids(src_layout: Path, 
                             beamid_by_ant: dict[int, int], 
                             beamvar_type: str = 'beamdflt',
                             beam_map_csv=None,
                             analytic_beam: tuple | None = None,
                             analytic_beam_map: tuple | None = None,  
                             ) -> Path:
    """
    Read a whitespace-separated array layout (header: Name Number [BeamID] E N U ...),
    inject/overwrite a BeamID column, and write a tab-separated copy.

    Returns the new path (src.with_suffix(src.suffix + '.with_beamids')).
    """
    lines = Path(src_layout).read_text().splitlines()
    if not lines:
        raise ValueError(f"{src_layout} is empty.")

    # Split on ANY whitespace to support tabs OR spaces
    header = re.split(r"\s+", lines[0].strip())

    musthave = {"Name", "Number", "E", "N", "U"}
    if not musthave.issubset(set(header)):
        raise ValueError(f"{src_layout} missing required columns {musthave}. Got: {header}")

    has_beamid = "BeamID" in header
    if has_beamid:
        beamid_idx = header.index("BeamID")
        new_header = header[:]  # overwrite in place later
    else:
        # insert BeamID right after Number
        num_idx = header.index("Number")
        beamid_idx = num_idx + 1
        new_header = header[:beamid_idx] + ["BeamID"] + header[beamid_idx:]

    # map header name -> index in the *original* file
    idx = {name: i for i, name in enumerate(header)}

    out_lines = []
    out_lines.append("\t".join(new_header))  # write as TSV

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = re.split(r"\s+", line.strip())

        # robustly get antenna Number (int), even if stored like "0.0"
        ant_num = int(float(parts[idx["Number"]]))
        if ant_num not in beamid_by_ant:
            raise ValueError(f"No BeamID mapping provided for antenna Number={ant_num}")

        beamid = str(beamid_by_ant[ant_num])

        if has_beamid:
            # overwrite existing BeamID column
            parts[beamid_idx] = beamid
            # ensure parts length matches header length; pad if needed
            if len(parts) < len(new_header):
                parts += [""] * (len(new_header) - len(parts))
            row = parts
        else:
            # insert BeamID right after Number
            row = parts[:beamid_idx] + [beamid] + parts[beamid_idx:]

        out_lines.append("\t".join(row))

    tag = ""
    # filename with beam_map tags
    if analytic_beam_map is not None:
        # Include beam class in filename for accessibility
        beam_dict = dict(analytic_beam_map)
        beam_tag = beam_dict['class'].replace('.', '_').replace('hera_sim_beams_', '')
        tag = f"{tag}_analyticmap_{beam_tag}"
    elif analytic_beam is not None:
        # Include beam class in filename for accessibility
        beam_dict = dict(analytic_beam)
        beam_tag = beam_dict['class'].replace('.', '_').replace('hera_sim_beams_', '')
        tag = f"{tag}_analytic_{beam_tag}"
    elif beam_map_csv is not None:
        h = hashlib.md5(Path(beam_map_csv).read_bytes()).hexdigest()[:8]
        tag = f"{tag}.{Path(beam_map_csv).stem}.{h}._beammapperant_{beamvar_type}"

    # dst = src_layout.with_suffix(src_layout.suffix + ".with_beamids")
    dst = src_layout.with_suffix(src_layout.suffix + f"{tag}.with_beamids")
    # print("printing layout file name" + dst)
    Path(dst).write_text("\n".join(out_lines) + "\n")
    return dst

@cache
def make_tele_config(
    freq_interp_kind: str = "cubic",
    spline_interp_order: int = 3,
    beam_interpolator: str = "az_za_map_coordinates",
    *,
    beam_map_csv: Path | None = None,
    default_beam_file: Path | None = None,
    beamvar_type: str = 'beamdflt',
    ideal_layout: bool = True,
    analytic_beam: tuple | None = None,
    analytic_beam_map: tuple | None = None,  # NEW: for multi-beam configs
) -> Path:
    """
    Make a telescope config file.

    If analytic_beam_map is provided: use multiple !AnalyticBeam entries
    If analytic_beam is provided: use single !AnalyticBeam tag
    If beam_map_csv is None : single-beam config default
    else : build mapping : BeamID : !UVBeam filename
    """
    # default beam 
    if default_beam_file is None:
        default_beam_file = Path(utils.BEAMDIR) / "NF_HERA_Vivaldi_efield_beam_extrap.fits"

    config = []
    
    # Multiple analytic beam input
        # Multi-analytic beam mode (NEW)
    if analytic_beam_map is not None:
        # Convert tuple back to dict
        beam_defs = {}
        for beam_id, params_tuple in analytic_beam_map:
            if beam_id == "_default_beam_id":
                continue
            params = dict(params_tuple)
            if "beam_coeffs" in params and isinstance(params["beam_coeffs"], tuple):
                params["beam_coeffs"] = list(params["beam_coeffs"])
            beam_defs[beam_id] = params
        
        config.append("beam_paths:")
        for beam_id in sorted(beam_defs.keys()):
            params = beam_defs[beam_id]
            config.append(f"  {beam_id}: !AnalyticBeam")
            for key, value in params.items():
                config.append(f"    {key}: {value}")

    #  Single analytical beam pathing
    elif analytic_beam is not None:
        config.append("beam_paths:")
        config.append("  0: !AnalyticBeam")
        config.append(f"    class: {analytic_beam['class']}")
        
        # Add all other parameters from the config dict
        for key, value in analytic_beam.items():
            if key == 'class':
                continue
            if isinstance(value, list):
                # Format list nicely for YAML
                config.append(f"    {key}: {value}")
            else:
                config.append(f"    {key}: {value}")

    # Per-antenna: build beam_paths from mapping
    elif beam_map_csv is not None:
        mapping = read_beam_map_csv(beam_map_csv)
        unique_files = {}
        for ant in sorted(mapping):
            fpath = mapping[ant]
            if fpath not in unique_files:
                unique_files[fpath] = len(unique_files)

        config.append("beam_paths:")
        # Write in BeamID order
        for f, bid in sorted(unique_files.items(), key=lambda kv: kv[1]):
            print(f"Writing beam : {bid}, '{str(f)}' ")
            config.append(f"  {bid}: !UVBeam")
            config.append(f"    filename: '{str(f)}'")

    else:
        # default beam
        config.append("beam_paths:")
        config.append("  0: !UVBeam")
        config.append(f"    filename: '{str(default_beam_file)}'")

    # telescope, interp meta
    config.append(f"telescope_location: {str(utils.HERA_LOC)}")
    config.append("telescope_name: HERA")
    config.append(f"freq_interp_kind: '{freq_interp_kind}'")

    # beam_interpolator type
    if beam_interpolator == "az_za_simple":
        config.append("spline_interp_opts:")
        config.append(f"  kx: {int(spline_interp_order)}")
        config.append(f"  ky: {int(spline_interp_order)}")
    elif beam_interpolator == "az_za_map_coordinates":
        config.append("spline_interp_opts:")
        config.append(f"  order: {int(spline_interp_order)}")

    # filename with beam_map tags
    tag = f"{freq_interp_kind}_{spline_interp_order}"
    if analytic_beam_map is not None:
        from hashlib import md5
        ideal_tag = "idealT" if ideal_layout else "idealF"
        map_hash = md5(str(analytic_beam_map).encode()).hexdigest()[:8]
        tag = f"{tag}_analyticmap_{ideal_tag}_{map_hash}"
    elif analytic_beam is not None:
        # Include beam class in filename for accessibility
        beam_dict = dict(analytic_beam)
        beam_tag = beam_dict['class'].replace('.', '_').replace('hera_sim_beams_', '')
        ideal_tag = "idealT" if ideal_layout else "idealF"
        tag = f"{tag}_analytic_{ideal_tag}_{beam_tag}"
    elif beam_map_csv is not None:
        bm_blob = Path(beam_map_csv).read_bytes()
        ideal_tag = "idealT" if ideal_layout else "idealF"
        tag = f"{tag}_beammapperant_{ideal_tag}_{beamvar_type}"

    _fname = f"hera_{tag}.yaml"
    fname = CFGDIR / "teleconfigs" / "tmp" / _fname
    fname.parent.mkdir(exist_ok=True, parents=True)
    Path(fname).write_text("\n".join(config) + "\n")
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
    prefix: str = "default",
    *,
    beam_map_csv = None,		#per antenna beamfile (explicit path pass)
    default_beam_file = None,		#default antenna beamfile (usually None since harcoded option in make_tele_config() to avoid breaking legacy code
    beamvar_type: str = 'beamdflt',
    analytic_beam: dict | None = None,
    analytic_beam_map_file: Path | None = None,  
    sky_realization: str | None = None,
):
    """Create obsparam files (one per channel × chunk)"""
    freq_vals = utils.FREQS_DICT[season][channels]

    if NTIMES % chunks != 0:
        raise ValueError(f"Please choose chunks to divide NTIMES {NTIMES} cleanly")

    print("chunks: ", chunks)
    if do_chunks is None:
        do_chunks = list(range(chunks + 1))
    else:
        assert all(x < chunks for x in do_chunks)
    print(do_chunks)
    Ntimes_per_chunk = NTIMES // chunks


    # Build the sky catalog path
    if sky_realization:
        sky_subdir = f"{sky_model}/{sky_realization}"
    else:
        sky_subdir = sky_model

#    if isinstance(layout, str):
#    	# it's a namey
#        layout_file = utils.make_hera_layout(name=layout, ideal=ideal_layout)
#    elif isinstance(layout, Path):
#        layout_file = layout
#    else:
#    	# it's a list of integers specifying antennas
#        layout_file = utils.make_hera_layout(
#            name=f"HERA_custom_subset_{md5(str(layout).encode()).hexdigest()}",
#            ants=layout,
#            ideal=ideal_layout,
#        )

    # print("Input layout is ", layout)
    # layout=[0,1,2,3,4,5,6,7,8,9]
    # print("Input layout is ", layout)
    if isinstance(layout, str):
        # It's a known layout name (must exist in ANTS_DICT)
        layout_file = utils.make_hera_layout(name=layout, ideal=ideal_layout)
        layout_key = layout  # for directory naming
    elif isinstance(layout, Path):
        layout_file = layout
        layout_key = layout.stem
    elif isinstance(layout, (list, tuple)):
        # Layout is explicitly a list of antenna numbers
        subset = list(map(int, layout))
        print("subset is ", subset)
        # This adds a tag to folder names to encode if ideal antpos used or non-red
        ideal_tag = "idealT" if ideal_layout else "idealF"
        subset_name = f"HERA_custom_subset_{ideal_tag}_{md5(str(subset).encode()).hexdigest()}"
        layout_file = utils.make_hera_layout(
            name=subset_name,
            ants=np.array(subset),
            ideal=ideal_layout,
        )
        layout_key = subset_name
    else:
        raise ValueError(
            f"Invalid layout argument: {layout!r}. "
            "Expected a layout name (str), a file (Path), or a list of antenna integers."
        )
        

    ###############################New Part############################################
    # If per-antenna beams provided, produce a copy of the layout with BeamID column
    # and remember the new path; otherwise keep legacy layout.
    analytic_beam_tuple = None
    # Multi-beam analytic mode
    if analytic_beam_map_file is not None:
        from .anabeam_config import read_analytic_beam_map
        beam_definitions, antenna_to_beamid = read_analytic_beam_map(analytic_beam_map_file)
        default_bid = beam_definitions.pop("_default_beam_id", 0)
        
        # Convert to hashable tuple for @cache
        beam_defs_hashable = []
        for bid, params in beam_definitions.items():
            params_tuple = tuple((k, tuple(v) if isinstance(v, list) else v) 
                                  for k, v in sorted(params.items()))
            beam_defs_hashable.append((bid, params_tuple))
        analytic_beam_map_tuple = tuple(sorted(beam_defs_hashable))
        
        # Read layout to get antenna numbers, assign default bid to unlisted
        with open(layout_file) as f:
            lines = f.readlines()
        header = lines[0].strip().split()
        num_idx = header.index("Number")
        
        beamid_by_ant = {}
        for line in lines[1:]:
            parts = line.strip().split()
            if parts:
                ant = int(parts[num_idx])
                beamid_by_ant[ant] = antenna_to_beamid.get(ant, default_bid)   
        # Write layout with BeamID column
        layout_file = write_array_with_beamids(layout_file, 
                                               beamid_by_ant,
                                               beamvar_type=beamvar_type,
                                               beam_map_csv=None,
                                               analytic_beam=None,
                                               analytic_beam_map=analytic_beam_map_tuple,
                                               )
        tele_config_file = make_tele_config(
            freq_interp_kind=freq_interp_kind,
            spline_interp_order=spline_interp_order,
            beam_interpolator=beam_interpolator,
            beam_map_csv=None,
            default_beam_file=None,
            beamvar_type=beamvar_type,
            ideal_layout=ideal_layout,
            analytic_beam=None,
            analytic_beam_map=analytic_beam_map_tuple,
        )
    elif analytic_beam is not None:
        # Convert nested lists to tuples for hashability
        analytic_beam_hashable = {}
        for k, v in analytic_beam.items():
            if isinstance(v, list):
                analytic_beam_hashable[k] = tuple(v)
            else:
                analytic_beam_hashable[k] = v
        analytic_beam_tuple = tuple(sorted(analytic_beam_hashable.items()))
        tele_config_file = make_tele_config(
            freq_interp_kind=freq_interp_kind,
            spline_interp_order=spline_interp_order,
            beam_interpolator=beam_interpolator,
            beam_map_csv=None,
            default_beam_file=None,
            beamvar_type=beamvar_type,
            ideal_layout=ideal_layout,
            analytic_beam=analytic_beam_tuple,
            analytic_beam_map=None,
        )
    elif beam_map_csv is not None:
        # Build mapping ant to BeamID
        mapping = read_beam_map_csv(beam_map_csv)
        # Assign IDs by first-appearance of unique files
        unique_files: dict[Path, int] = {}
        beamid_by_ant: dict[int, int] = {}
        for ant in sorted(mapping):
            f = mapping[ant]
            if f not in unique_files:
                unique_files[f] = len(unique_files)
            beamid_by_ant[ant] = unique_files[f]
        # Write layout copy with BeamID column
        layout_file = write_array_with_beamids(layout_file, 
                                               beamid_by_ant, 
                                               beamvar_type=beamvar_type,
                                               beam_map_csv=beam_map_csv,
                                               analytic_beam=None,
                                               analytic_beam_map=None,
                                               )
        # Write a telescope config with beam_paths for all IDs
        tele_config_file = make_tele_config(
            freq_interp_kind=freq_interp_kind,
            spline_interp_order=spline_interp_order,
            beam_interpolator=beam_interpolator,
            beam_map_csv=beam_map_csv,
            default_beam_file=default_beam_file,
            beamvar_type=beamvar_type,
            ideal_layout=ideal_layout,
            analytic_beam=None,
            analytic_beam_map=None,
        )
    else:
        tele_config_file = make_tele_config(
            freq_interp_kind=freq_interp_kind,
            spline_interp_order=spline_interp_order,
            beam_interpolator=beam_interpolator,
            beam_map_csv=None,
            default_beam_file=default_beam_file,
            beamvar_type=beamvar_type,
            ideal_layout=ideal_layout,
            analytic_beam=None,
            analytic_beam_map=None,
        )
    ####################################################################################

    sky_model_key = f"{sky_model}/{sky_realization}" if sky_realization else sky_model
    print("sky_model_key is ", sky_model_key)

    modeldir = utils.get_direc(
        sky_model=sky_model_key,
        chunks=chunks,
        layout=layout_file.stem,
        redundant=redundant,
        prefix=prefix,
        beamvar_type=beamvar_type,
    )
    
    print("out_file.stem is ", layout_file.stem)
    print("out_file is ", layout_file)

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

            ants = np.genfromtxt(layout_file, skip_header=1, usecols=(1, 3, 4, 5), delimiter="\t")
            antnums = ants[:, 0]
            redbls = get_antenna_redundancies(antnums, ants[:, 1:], tol=4.0, use_grid_alg=True, include_autos=True)[0]  # hera thresh
            redbls = np.array([baseline_to_antnums(r[0], Nants_telescope=350) for r in redbls])
            np.savetxt(redfile, redbls)
        reds = [(int(a), int(b)) for a, b in redbls]

    print(channels, freq_vals, do_chunks)
    for fch, fv in zip(channels, freq_vals):
        print("test 2")
        for ch in do_chunks:
            print("test 3")
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
                    "channel_width": float(utils.FREQS_DICT[season][1] - utils.FREQS_DICT[season][0]),
                    "start_freq": float(fv),
                },
                "sources": {"catalog": f"{SKYDIR}/{sky_subdir}/fch{fch:04d}.skyh5"},
                "telescope": {
                    "array_layout": f"{layout_file}",
                    "telescope_config_name": f"{tele_config_file}",
                    "select": {"freq_buffer": 3.0e6},
                },
                "time": {
                    "Ntimes": Ntimes_per_chunk,
                    "integration_time": INTEGRATION,
                    "start_time": START_TIME + INTEGRATION * ch * Ntimes_per_chunk / 86400,
                },
                "polarization_array": [-5, -7, -8, -6],
                "cat_name": sky_model,
            }

            if redundant:
                # top-level selection of redundant baselines
                obsparams["select"] = {"bls": str(reds)}

            with open(obsparams_file, "w") as stream:
                yaml.dump(obsparams, stream, default_flow_style=False, sort_keys=False)

            print(f"Wrote obsparams at {obsparams_file}")

    print("test 1")

    return layout_file
