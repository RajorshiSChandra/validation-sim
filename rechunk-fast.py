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
import os
import logging 
from hera_cal import io
from collections import Counter

import h5py
import numpy as np
import psutil
from pyuvdata.uvdata.uvh5 import FastUVH5Meta
from pyuvdata import utils as uvutils
from pathlib import Path
from hera_cal._cli_tools import parse_args, run_with_profiling
from multiprocessing import shared_memory, Pool, cpu_count
from functools import partial

logger = logging.getLogger('rechunk')
ps = psutil.Process()

def find_all_files(
        base_dir: Path,
        channels: list[int],
        prototype: str,
        ignore_missing_channels: bool = False,
        assume_blt_layout: bool = False,
        is_rectangular: bool | None = None,
):
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
    fl0 = FastUVH5Meta(all_files[channels[0]][0], blts_are_rectangular=is_rectangular)
    
    if not fl0.blts_are_rectangular:
        raise ValueError("Your first file is not rectangular. This script only works for rectangular files...")
        
    for ch in channels:
        all_files[ch] = [FastUVH5Meta(
            fl,
            blts_are_rectangular=fl0.blts_are_rectangular if assume_blt_layout else None, 
            time_first=fl0._time_first if assume_blt_layout else None
        ) for fl in all_files[ch]]

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
            time_index = np.argwhere(meta.lsts >= lst_wrap).flatten()[0]
            break

    Ntimes = meta_list[0].Ntimes

    total_times = Ntimes * len(meta_list)
    logger.info(f"Total times: {total_times}")
    nchunks = int(np.ceil(total_times / lsts_per_chunk))
    logger.info(f"Total chunks: {nchunks}")
    chunks = [[] for _ in range(nchunks)]
    
    for chunk in chunks:
        times_remaining = lsts_per_chunk

        while True:
            chunk.append((
                fl_index,
                slice(time_index, min(Ntimes, time_index + times_remaining)) 
            ))

            if time_index + lsts_per_chunk >= Ntimes:
                # We have to use the start of the next file
                times_remaining -= (Ntimes - time_index)
                time_index = 0
                fl_index += 1
                if fl_index >= len(meta_list):
                    fl_index = 0
            else:
                times_remaining = 0
                time_index += lsts_per_chunk

            if times_remaining == 0:
                break

    logger.debug(f"CHUNKS: {chunks}")
    return chunks

def reset_time_arrays(uvd, meta, times, lsts, ras, pas, slc, time_first):
    nt = slc.stop - (slc.start or 0)

    if nt != uvd.Ntimes:
        uvd.Nblts = uvd.Nbls * nt
        uvd.integration_time = uvd.integration_time[0] * np.ones(uvd.Nblts)
        uvd.phase_center_app_dec = uvd.phase_center_app_dec[0] * np.ones(uvd.Nblts)
        uvd.phase_center_id_array = np.zeros(uvd.Nblts, dtype=int)
    
    if time_first:
        uvd.time_array = np.tile(times[slc], uvd.Nbls)
        uvd.phase_center_app_ra = np.tile(ras[slc], uvd.Nbls)
        uvd.phase_center_frame_pa = np.tile(pas[slc], uvd.Nbls)
        uvd.lst_array = np.tile(lsts[slc], uvd.Nbls)
        if nt != uvd.Ntimes:
            uvd.uvw_array = np.repeat(uvd.uvw_array[::meta.Ntimes], nt, axis=0)
            uvd.ant_1_array = np.repeat(meta.unique_ant_1_array, nt)
            uvd.ant_2_array = np.repeat(meta.unique_ant_2_array, nt)
            uvd.baseline_array = np.repeat(meta.unique_baseline_array, nt)
    else:
        uvd.time_array = np.repeat(times[slc], uvd.Nbls)
        uvd.phase_center_app_ra = np.repeat(ras[slc], uvd.Nbls)
        uvd.phase_center_frame_pa = np.repeat(pas[slc], uvd.Nbls)
        uvd.lst_array = np.repeat(lsts[slc], uvd.Nbls)
        if nt != uvd.Ntimes:
            uvd.uvw_array = np.tile(uvd.uvw_array[:meta.Nbls], (nt,1))
            uvd.ant_1_array = np.tile(meta.unique_ant_1_array, nt)
            uvd.ant_2_array = np.tile(meta.unique_ant_2_array, nt)
            uvd.baseline_array = np.tile(meta.unique_baseline_array, nt)

    uvd.Ntimes = nt

def chunk_files(
    channels, r_prototype: str, save_dir: Path, base_dir: Path, 
    prototype: str, lst_wrap: float = np.pi, n_times_per_file: int = 180, 
    ignore_missing_channels: bool=False, assume_blt_layout: bool = False,
    blt_order ='determine',
    max_mem_mb: int = 100000,
    nthreads: int | None = None,
    is_rectangular: bool | None = None,
):
    # Load the read files, and check that the read prototype is valid if provided.
    raw_files = find_all_files(base_dir, channels, r_prototype, ignore_missing_channels, assume_blt_layout, is_rectangular=is_rectangular)
    
    logger.info(f"Number of raw data files: {len(raw_files)}")

    # Ensure all the files have rectangular blts with the same ordering.
    # This is a requirement for the chunking to work.
    time_first = raw_files[channels[0]][0]._time_first
    if not assume_blt_layout:
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

    if lst_wrap is None:
        lst_wrap = np.min(lsts)

    # Roll the times and lsts around so it's easier to chunk them.
    lsts[lsts < lst_wrap] += 2*np.pi
    time_index = np.argwhere(lsts >= lst_wrap)[0][0]
    times = np.roll(times, -time_index)
    lsts = np.roll(lsts, -time_index) % (2*np.pi)

    # Make a prototype UVData object for the chunked data.
    uvd = meta.to_uvdata()  # this has too many times, and only one frequency. We update that manually.
    uvd.use_future_array_shapes()
    uvd.freq_array = freqs
    uvd.Nfreqs = len(freqs)
    uvd.channel_width = np.ones_like(freqs)*(np.diff(freqs)[0])
    
    if time_first:
        phase_center_ra = uvd.phase_center_app_ra[:meta.Ntimes]
        phase_center_pa = uvd.phase_center_frame_pa[:meta.Ntimes]
    else:
        phase_center_ra = uvd.phase_center_app_ra[::meta.Nbls]
        phase_center_pa = uvd.phase_center_frame_pa[::meta.Nbls]

    phase_center_ra = np.roll(phase_center_ra, -time_index)
    phase_center_pa = np.roll(phase_center_pa, -time_index)

    logger.info(f"Data has {uvd.Nbls} baselines and {uvd.Ntimes} times per file")

    # Get the slices we'll need for each chunk.
    logger.info("Getting time slices for each output file...")
    chunk_slices = get_file_time_slices(raw_files[channels[0]], n_times_per_file, lst_wrap)
    logger.info("Got all time slices")
    
    
    #metas0 = raw_files[channels[0]]

    # Figure out how many frequencies we can fit in the data at once.
    current_mem = ps.memory_info().rss
    mem_left = max_mem_mb*(1024**2) - current_mem - 100*(1024**2)  # leave 100MB for overhead
    mem_per_freq = (n_times_per_file * uvd.Nbls * uvd.Npols * 8)
    nfreqs = min(mem_left // mem_per_freq, uvd.Nfreqs)

    nfreq_chunks = int(np.ceil(uvd.Nfreqs / nfreqs))
    logger.info(f"Going to use {nfreq_chunks} frequency chunks of {nfreqs} frequencies each.")
    logger.info(f"This is estimated to use {mem_per_freq*nfreqs/1024**2:.2f} MB of memory (of the {mem_left/1024**2} MB left).")
    logger.info("")
    for outfile_index, chunk_slices in enumerate(chunk_slices):
        logger.info(f"Creating data for file {outfile_index + 1}")
        # Update the metadata for this chunk.
        time_slice = slice(outfile_index*n_times_per_file, min(len(times), (outfile_index+1)*n_times_per_file))
        reset_time_arrays(uvd, meta, times=times, lsts=lsts, ras=phase_center_ra, pas=phase_center_pa, time_first=time_first, slc=time_slice)

        # Check on first, second and last.
        if outfile_index in [0, 1, len(chunk_slices)-1]:
            uvd.check()  # quick check to make sure we set everything up correctly.

        fname = prototype.format(lst=uvd.lst_array[0])
        pth = save_dir / fname
        
        # This just writes the header.
        logger.info("Initializing UVH5 file...")
        uvd.initialize_uvh5_file(pth, clobber=True)

        pool = Pool(nthreads or cpu_count())
        raw_file_paths = {ch: [f.path for f in raw_files[ch]] for ch in channels}

        for freq_chunk in range(nfreq_chunks):
            logger.info(f"Obtaining frequency chunk {freq_chunk+1}/{nfreq_chunks}...")
            freq_slice = slice(freq_chunk * nfreqs, min((freq_chunk+1)*nfreqs, uvd.Nfreqs))
            this_nfreq = freq_slice.stop - freq_slice.start
            SHAPE = (uvd.Nblts, this_nfreq, uvd.Npols)
#            full_dset = np.empty(, dtype=np.complex64)

            shm = shared_memory.SharedMemory(create=True, size=np.dtype(np.complex64).itemsize * np.prod(SHAPE), name='FULLDSET')
            full_dset = np.ndarray(SHAPE, dtype=np.complex64, buffer=shm.buf)

            # Now we need to actually write the data.
            
            pool.map(
                partial(
                    write_freq_chunk, ntimes=uvd.Ntimes, chunk_slices=chunk_slices, 
                    nbls=uvd.Nbls, raw_files=raw_file_paths, time_first=time_first, 
                    shape=SHAPE, channels=channels
                ), 
                range(freq_slice.start, freq_slice.stop)
            )
            shm.unlink()
            # Now write the data.
            with h5py.File(pth, 'a') as fl:
                fl['/Data/visdata'][:, freq_slice] = full_dset
                del full_dset

def write_freq_chunk(ich: int, ntimes, chunk_slices, nbls, raw_files, time_first, shape, channels):
    ntimes_left = ntimes
    nblts_so_far = 0
    ch = channels[ich]

    shm = shared_memory.SharedMemory(name='FULLDSET')
    full_dset = np.ndarray(shape, dtype=np.complex64, buffer=shm.buf)

    for (flidx, slc) in chunk_slices:
        if ntimes_left == 0:
            break

        this_ntimes = min(slc.stop - slc.start, ntimes_left)
        this_nblts = this_ntimes * nbls
        
        # Get the data for this chunk.
        pth = raw_files[ch][flidx]
        with h5py.File(pth, 'r') as fl:
            if time_first:
                slices = [slice(slc.start + n, slc.start + this_ntimes + n) for n in range(nbls)]
            else:
                slices = slice(nbls * slc.start, nbls * (slc.start + this_ntimes))
        
            data = fl['/Data/visdata'][slices]
        

        full_dset[nblts_so_far:nblts_so_far + this_nblts, ich] = data[:this_nblts, 0]
        
        ntimes_left -= this_ntimes
        nblts_so_far += this_nblts
    
    shm.close()

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
        "--lst-wrap", type=float, default=None, help="Where to perform the wrap in LST. Default is lowest LST in data."
    )
    parser.add_argument(
        "--clobber", default=False, action="store_true", help="Overwrite existing files."
    )
    parser.add_argument(
        "--ignore-missing-channels", action='store_true', help='merely warn if there are no files for a particular channel'
    )
    parser.add_argument(
        "--assume-same-blt-layout", action='store_true', help='whether to assume each file has the same layout of baselines/times'
    )
    parser.add_argument(
        "--is-rectangular", action='store_true', help='whether blts are rectangular', 
    )
    parser.add_argument(
        "--max-mem", type=int, default=1e9, help="Maximum memory to use in MB."
    )
    parser.add_argument(
        "--nthreads", type=int, default=None, help="Number of threads to use."
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

    # Build the save file prototype.
    if args.sky_cmp:
        prototype = "zen.LST.{lst:.7f}." + f"{args.sky_cmp}.uvh5"
    else:
        prototype = "zen.LST.{lst:.7f}.uvh5"
    

    channels = sum(
        (list(range(*tuple(map(int, ch.split("~"))))) for ch in args.channels),
        start=[],
    )

    run_with_profiling(
        chunk_files, args,
        prototype=prototype,
        channels=channels,
        lst_wrap=args.lst_wrap,
        n_times_per_file=args.chunk_size,
        save_dir=save_dir, base_dir=base_dir,
        r_prototype=args.r_prototype,
        ignore_missing_channels=args.ignore_missing_channels,
        assume_blt_layout=args.assume_same_blt_layout,
        is_rectangular=args.is_rectangular,
        max_mem_mb=args.max_mem,
        nthreads=args.nthreads,
    )
