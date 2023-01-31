"""Script for taking per-frequency simulation data and chunking it in time.

Output files have the following prototype:
    zen.LST.{lst:.7f}[.{sky_cmp}].uvh5

Since we're interested in matching observations by LST, it makes more sense
to group files by the LSTs they cover.

Some reference info: chunked files containing 180 times and the full array take
up about 13 GB of disk space. If this is the true size of the chunked data in
memory, then we can load in roughly 10 times the times at once and chunk those.
Note that chunking files into groups of 180 times gives us 96 chunks. So the thing
to try in order to speed this process up is to load in chunks of 2160 times, inflate,
then chunk those into 180 integration files.

Note that the input files may be partial in frequency and time.

Note that it takes a little over 3 hours of wall time in order to make a single 180
integration chunk using the full band.
"""
import argparse
import copy
import os
import time
import logging 
from hera_cal import io

import numpy as np
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from pyuvsim.simsetup import _parse_layout_csv
from pathlib import Path
from hera_cal._cli_utils import parse_args, run_with_profiling


def find_all_files(base_dir: Path, channels: list[int], prototype: str, ignore_missing_channels: bool = False):
    all_files = {}
    for ch in channels:
        all_files[ch] = {}
        p = prototype.format(channel=ch)
        files = list(base_dir.glob(p))
        if not files and not ignore_missing_channels:
            raise FileNotFoundError(f"No files with prototype {p} for channel {ch}")
        
def chunk_files(args):
    # Load the read files, and check that the read prototype is valid if provided.
    
    raw_files = sorted(list(base_dir.glob(args.r_prototype)))
    if len(raw_files) == 0:
    raise FileNotFoundError(
        f"No files with the given prototype '{args.r_prototype}' were "
        f"found in the provided base directory '{args.base_dir}'."
    )
    
    logging.info(f"Number of raw data files: {len(raw_files)}")

    # Build the save file prototype.
    if args.sky_cmp:
        prototype = "zen.LST.{lst:.7f}." + f"{args.sky_cmp}.uvh5"
    else:
        prototype = "zen.LST.{lst:.7f}.uvh5"
    
    # Read the metadata from a file to get the times.
    logging.info("Reading reference metadata...")
    t1 = time.time()
    uvd = UVData()
    uvd.read(raw_files[0], read_data=False)
    time_array, inds, inv_inds = np.unique(
        uvd.time_array, return_index=True, return_inverse=True
    )
    lsts = uvd.lst_array[inds]
    Ntimes = uvd.Ntimes
    dt = time.time() - t1
    logging.info(f"Finished in {dt} seconds.")

    # Check that the array layout file, if provided, exists.
    if args.array_layout:
        if not os.path.exists(args.array_layout):
            raise FileNotFoundError(
                f"The provided path to the array layout, {args.array_layout}, "
                "does not exist."
            )

        logging.info(f"Loading array data...")
        t1 = time.time()
        # This gives an array of 6-tuples: (name, num, beam_id, e, n, u)
        array_info = _parse_layout_csv(args.array_layout)
        
        # Update the values of the antenna information parameters for later.
        antenna_names = np.array(
            [item[0] for item in array_info], dtype=str
        )
        antenna_numbers = np.array(
            [item[1] for item in array_info], dtype=uvd._antenna_numbers.value.dtype
        )
        antenna_diameters = np.array(
            [uvd._antenna_diameters.value[0],] * len(array_info),
            dtype=uvd._antenna_diameters.value.dtype
        )
        antenna_positions = np.array(
            [list(item)[3:] for item in array_info], dtype=uvd._antenna_positions.value.dtype
        )
        # Change antenna positions from ENU to ECEF.
        antenna_positions = uvutils.ECEF_from_ENU(
            antenna_positions, *uvd.telescope_location_lat_lon_alt
        ) - uvd.telescope_location
        Nants_telescope = len(array_info)
        dt = time.time() - t1
        logging.info(f"Finsihed in {dt} seconds.")

    # Here's the idea: the labels will give the LSTs on [lst_wrap,lst_wrap+2pi), and
    # when the daily data is made, we'll put the LSTs into this space and do the
    # interpolation that way. It should work fine, since the wrapping point is chosen
    # so that it's the first LST of the first day and just after the last LST of the
    # last day.
    # The wrap index selection is a little sketchy in general, but will work fine for
    # H1C IDR3 validation.
    wrap_ind = np.argwhere(lsts > args.lst_wrap).flatten()[0]
    # wrap_ind = np.argmin(np.abs(lsts - args.lst_wrap))
    new_lsts = np.roll(lsts, -wrap_ind)
    new_lsts[new_lsts < new_lsts[0]] += 2 * np.pi
    one_day = np.mean(np.diff(time_array)) * Ntimes
    new_times = np.roll(time_array, -wrap_ind)
    lst_wrap = new_lsts[0]

    # Read in the big chunk all at once to save on IO costs.
    start = args.n_stride * args.n_times_to_load
    end = (1 + args.n_stride) * args.n_times_to_load
    start_lsts = new_lsts[start:end:args.chunk_size]
    times_to_load = new_times[start:end]
    lsts_to_load = new_lsts[start:end] % (2 * np.pi)
    iswrapped = lsts_to_load[0] > lsts_to_load[-1]
    del uvd
    logging.info(f"Number of time chunks to produce: {len(start_lsts)}")
    logging.info(f"List of starting LSTs: {start_lsts}")

    # Actually read in the files. Need to do it this way to deal with metadata issues.
    logging.info("Reading raw simulation files...")
    t1 = time.time()
    new_uvd = UVData()
    new_uvd.read(raw_files, times=times_to_load, axis="freq")
    new_uvd.x_orientation = "east"
    dt = (time.time() - t1) / 3600
    logging.info(f"Took {dt:.4f} hours to read all files.")

    if iswrapped:
        # Need to re-sort since select sorts by increasing time.
        blt_inds = np.concatenate(
            [np.argwhere(new_uvd.time_array == t).flatten() for t in times_to_load]
        )
        new_uvd.reorder_blts(blt_inds)

    # Modify the LST/time array if necessary.
    to_adjust = new_uvd.lst_array < lst_wrap
    if np.any(to_adjust):
        new_uvd.time_array[to_adjust] += one_day
        times_to_load[lsts_to_load < lst_wrap] += one_day

    # If an array layout is provided, then we need to update the metadata.
    if args.array_layout:
        logging.info("Updating array information...")
        # Update the array layout information.
        new_uvd.antenna_numbers = antenna_numbers
        new_uvd.antenna_names = antenna_names
        new_uvd.antenna_positions = antenna_positions
        new_uvd.antenna_diameters = antenna_diameters
        new_uvd.Nants_telescope = Nants_telescope
        if args.compress:
            new_uvd.compress_by_redundancy()
        elif args.inflate:
            new_uvd.inflate_by_redundancy()

    if args.compress and not args.array_layout:
        new_uvd.compress_by_redundancy()

    logging.info("Writing files to disk...")
    t1 = time.time()
    if len(start_lsts) > 1:
        for chunk, start_lst in enumerate(start_lsts):
            # Figure out which times to select.
            start = chunk * args.chunk_size
            end = start + args.chunk_size
            times_here = times_to_load[start:end]
            uvd_here = new_uvd.select(times=times_here, inplace=False)

            # We should be done, so write the contents to disk.
            new_filename = save_dir / prototype.format(lst=start_lst)
            uvd_here.write_uvh5(new_filename, clobber=args.clobber)
            del uvd_here
    else:  # We don't want to make a copy of the data if we're only doing one chunk.
        new_filename = save_dir / prototype.format(lst=start_lsts[0])
        new_uvd.write_uvh5(new_filename, clobber=args.clobber)
    dt = time.time() - t1
    logging.info(f"Finished after {dt} seconds.")
    runtime = (time.time() - init_time) / 3600
    logging.info(f"Entire script took {runtime:.4f} hours to complete.")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dir", type=str, help="Path to directory containing simulation data."
    )
    parser.add_argument(
        "save_dir", type=str, help="Path to where to write new files."
    )
    parser.add_argument(
        "-r", "--r-prototype", type=str, default="", help="glob-parsable prototype of files to read. Can include a {channel} format string which will be replaced by the channel internally."
    )
    parser.add_argument(
        "--channels", type=str, help='the channels to read, in the form "low~high", eg "0~1535"', nargs="+"
    )
    parser.add_argument(
        "-s", "--sky-cmp", type=str, default=None, help="Sky component (e.g. diffuse)."
    )
    
    parser.add_argument(
        "--chunk-size", type=int, default=180, help="Number of integrations per chunk."
    )
    parser.add_argument(
        "--lst-wrap", type=float, default=np.pi, help="Where to perform the wrap in LST."
    )
    parser.add_argument(
        "--clobber", default=False, action="store_true", help="Overwrite existing files."
    )
    parser.add_argument(
        "--ignore-missing-channels", action='store_true', help='merely warn if there are no files for a particular channel'
    )

    args = parse_args(parser)
    
    # Check that the read/write directories actually exist with proper permissions.
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"The provided base directory '{args.base_dir}' does not exist."
        )

    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(
            "The provided save directory does not exist."
        )

    if not os.access(save_dir, os.W_OK):
        # Test that we have write privileges here to save time.
        raise PermissionError(
            "You do not have permission to write to the provided save "
            "directory. Please update the directory choice or permissions."
        )

    channels = sum(list(range(*tuple(map(int, ch.split("~"))))) for ch in args.channels, start=[])
    
    run_with_profiling(chunk_files, args, channels)

    
