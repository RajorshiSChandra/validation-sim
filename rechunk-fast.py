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
from collections import Counter

import h5py
import numpy as np
from pyuvdata import FastUVH5Meta
from pathlib import Path
from hera_cal._cli_utils import parse_args, run_with_profiling

logger = logging.getLogger('rechunk')

def find_all_files(base_dir: Path, channels: list[int], prototype: str, ignore_missing_channels: bool = False):
    all_files = {}
    for ch in channels:
        all_files[ch] = {}
        p = prototype.format(channel=ch)
        files = sorted(base_dir.glob(p))
        if not files:
            msg=f"No files with prototype {p} for channel {ch}"
            if not ignore_missing_channels:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
        else:
            all_files[ch] = files

    if not all_files:
        raise FileNotFoundError(f"No files found with prototype {prototype} in {base_dir}")

    nchunks_counter = Counter([len(flist) for flist in all_files.values()])
    if len(nchunks_counter) > 1:
        most_common_nchunks = nchunks_counter.most_common(1)[0][0]
        for fllist in all_files.values():
            if len(fllist) != most_common_nchunks:
                logger.warning(
                    "Number of files for different channels is not the same. "
                    f"Got {len(fllist)} for channel {ch} and {len(all_files[channels[0]])} for channel {channels[0]}"
                )

    for nc, count in nchunks_counter.items():
        logger.info(f"Found {count} channels with {nc} chunks")

    # Turn them all into metadata objects.
    for ch in channels:
        all_files[ch] = [FastUVH5Meta(fl) for fl in all_files[ch]]

    return all_files

def get_file_time_slices(meta_list: list[FastUVH5Meta], lsts_per_chunk: int,  lst_wrap: float):    
    dlst = -1
    i = 0
    while dlst < 0:
        dlst = meta_list[0].lsts[i+1] - meta_list[0].lsts[i]
        i += 1

    for fl_index, meta in enumerate(meta_list):
        if meta.lsts[0] > (lst_wrap + dlst):
            continue
        elif meta.lsts[-1] < lst_wrap:
            continue
        else:
            time_index = np.argwhere(meta.lsts > lst_wrap).flatten()[0]
            break

    starting_file = fl_index
    starting_time = time_index

    Ntimes = meta_list[0].Ntimes
    chunks = []
    while True:
        chunk = []
        ntimes_in_this_chunk = 0
        while True:
            chunk.append((
                fl_index,
                meta_list[fl_index],   # The meta object itself
                slice(time_index, min(Ntimes, time_index + lsts_per_chunk)) 
            ))

            if time_index + lsts_per_chunk >= Ntimes:
                ntimes_in_this_chunk += Ntimes - time_index
                time_index = time_index + lsts_per_chunk - Ntimes
                fl_index += 1
                if fl_index >= len(meta_list):
                    fl_index = 0
            else:
                ntimes_in_this_chunk += lsts_per_chunk

            if ntimes_in_this_chunk >= lsts_per_chunk:
                break
        
        if fl_index == starting_file and 0 < time_index - starting_time < lsts_per_chunk:
            break

    return chunks


def chunk_files(args, channels, save_dir: Path, base_dir: Path):
    # Load the read files, and check that the read prototype is valid if provided.
    raw_files = find_all_files(base_dir, channels, args.r_prototype, args.ignore_missing_channels)    
    
    logger.info(f"Number of raw data files: {len(raw_files)}")

    # Build the save file prototype.
    if args.sky_cmp:
        prototype = "zen.LST.{lst:.7f}." + f"{args.sky_cmp}.uvh5"
    else:
        prototype = "zen.LST.{lst:.7f}.uvh5"
    

    # Ensure all the files have rectangular blts with the same ordering.
    # This is a requirement for the chunking to work.
    time_first = raw_files[channels[0]][0]._time_first
    for ch in channels:
        for meta in raw_files[ch]:
            if not meta.blts_are_rectangular:
                raise ValueError("Not all files have rectangular blts.")
            if meta._time_first != time_first:
                raise ValueError("Not all files have the same blt ordering.")

    # Read the metadata from a file to get the times.
    logging.info("Reading reference metadata...")
    
    # Get all the frequencies we're gonna use.
    freqs = []
    for ch in channels:
        meta = raw_files[ch][0]
        freqs.append(meta.freqs)
    freqs = np.concatenate(freqs)

    # Get the times we're gonna use.
    times = []
    lsts = []
    for meta in raw_files[channels[0]]:
        times.append(meta.times)
        lsts.append(meta.lsts)
    times = np.concatenate(times)
    lsts = np.concatenate(lsts)

    # Roll the times and lsts around so it's easier to chunk them.
    time_index = np.argwhere(lsts >= args.lst_wrap)[0][0]
    times = np.roll(times, -time_index)
    lsts = np.roll(lsts, -time_index)


    n_chunks = int(np.ceil(len(lsts) / args.n_times_per_file))

    # Make a prototype UVData object for the chunked data.
    uvd = meta.to_uvd()  # this has too many times, and only one frequency. We update that manually.
    uvd.use_future_array_shapes()
    uvd.freq_array = freqs
    uvd.Nfreqs = len(freqs)
    uvd.channel_width = np.diff(freqs)[0]
    uvd.Nblts = uvd.Nbls * args.n_times_per_file
    uvd.Ntimes = args.n_times_per_file
    
    if time_first:
        uvd.time_array = np.tile(times[:args.n_times_per_file], uvd.Nbls)
        uvd.lst_array = np.tile(lsts[:args.n_times_per_file], uvd.Nbls)
        uvd.uvw_array = np.repeat(meta.uvw_array[::meta.Ntimes], uvd.Ntimes)
        uvd.ant_1_array = np.repeat(meta.unique_ant_1_array, args.n_times_per_file)
        uvd.ant_2_array = np.repeat(meta.unique_ant_2_array, args.n_times_per_file)
    else:
        uvd.time_array = np.repeat(times[:args.n_times_per_file], uvd.Nbls)
        uvd.lst_array = np.repeat(lsts[:args.n_times_per_file], uvd.Nbls)
        uvd.uvw_array = np.tile(meta.uvw_array[::meta.Ntimes], uvd.Ntimes)
        uvd.ant_1_array = np.tile(meta.unique_ant_1_array, args.n_times_per_file)
        uvd.ant_2_array = np.tile(meta.unique_ant_2_array, args.n_times_per_file)

    uvd.check()  # quick check to make sure we set everything up correctly.

    # Get the slices we'll need for each chunk.
    chunk_slices = get_file_time_slices(raw_files[channels[0]], args.n_times_per_file, args.lst_wrap)

#    metas0 = raw_files[channels[0]]

    
    for outfile_index, chunk_slices in enumerate(chunk_slices):
        
        # Update the metadata for this chunk.
        time_slice = slice(outfile_index*args.n_times_per_file, (outfile_index+1)*args.n_times_per_file)
        if time_first:
            uvd.time_array = np.tile(times[time_slice], uvd.Nbls)
            uvd.lst_array = np.tile(lsts[time_slice], uvd.Nbls)
        else:
            uvd.time_array = np.repeat(times[time_slice], uvd.Nbls)
            uvd.lst_array = np.repeat(lsts[time_slice], uvd.Nbls)
        
        fname = prototype.format(lst=uvd.lst_array[0])
        pth = save_dir / fname
        
        # This just writes the header.    
        uvd.write_uvh5(str(pth), clobber=True)

        full_dset = np.empty((uvd.Nblts, uvd.Nfreqs, uvd.Npols), dtype=np.complex64)

        # Now we need to actually write the data.
        for ich, ch in enumerate(channels):
            nblts_so_far = 0
            for (flidx, meta0, slc) in chunk_slices:
                # Get the data for this chunk.
                meta = raw_files[ch][flidx]
                with h5py.File(str(meta.path), 'r') as fl:
                    if time_first:
                        slices = [slice(slc.start + n, slc.stop + n) for n in range(uvd.Nbls)]        
                    else:
                        slices = slice(uvd.Nbls * slc.start, uvd.Nbls * slc.stop)
                data = fl['Data/visdata'][slices]
                
                full_dset[nblts_so_far:(data.shape[0] + nblts_so_far), ich] = data[:, 0]
                nblts_so_far += data.shape[0]
        
        # Now write the data.
        with h5py.File(pth, 'w') as fl:
            d = fl.create_group('Data')
            d['visdata'] = full_dset

    
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

    channels = sum(
        (list(range(*tuple(map(int, ch.split("~"))))) for ch in args.channels),
        start=[],
    )

    run_with_profiling(chunk_files, args, channels=channels, save_dir=save_dir, base_dir=base_dir)
