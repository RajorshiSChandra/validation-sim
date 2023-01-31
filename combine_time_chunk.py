from pathlib import Path
import argparse
from multiprocessing import Pool

import numpy as np
from pyuvdata import UVData
from hera_cal import io

VIS_DIR = Path(__file__).parent.absolute() / 'outputs'


def concat_blt(i):
    files_to_load = data_files_per_freq[i]
    outstem = data_files_per_freq[i][0].stem.replace('_chunk0', '')
    outfile = out_path / f'{outstem}.uvh5'
    
    print(f'Combining\n{files_to_load}\nWriting to {outfile}\n')
    
    print("Reading meta...")
    uvd = io.HERADataFastReader(files_to_load)
    print("Reading data")
    data = uvd.read(read_data=True, read_flags=False, read_nsamples=False)
    print("writing...")
    io.write_vis(
        outfile,
        data=data,
        lst_array=uvd.lsts,
        freq_array=uvd.freqs,
        antpos=uvd.antpos,
        time_array=uvd.times,
        filetype='uvh5',
        write_file=True,
        overwrite=True,
        verbose=True,
        history="Chunked by the chunker",
        object_name=args.sky_model,
        vis_units='Jy',
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine time-chunked visibility into single file'
    )
    parser.add_argument('sky_model', type=str)
    parser.add_argument('--ncores', type=int, default=1,
                        help='number of processes.')
    parser.add_argument('--channels', type=int, nargs='+', default=None,
                        help='channels to combine')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='output directory. default is same as inputs.')
    args = parser.parse_args()
    
    data_path = VIS_DIR / args.sky_model / 'nt17280'
    if args.out_dir is not None:
        out_path = Path(args.out_dir)
    else:
        out_path = data_path
    
    if args.channels is not None:
        data_files = []
        for ch in args.channels:
            data_files += sorted(
                data_path.glob(
                    f'{args.sky_model}_fch{ch:04d}_nt17280_chunk?.uvh5'
                )
            )
    else:  # gather all files
        data_files = sorted(
            data_path.glob(f'{args.sky_model}_fch????_nt17280_chunk?.uvh5')
        )

    channels = sorted({int(fl.name.split("_fch")[1].split("_")[0]) for fl in data_files})

    # Get the number of time chunks
    for i, fl in enumerate(data_files):
        if f'fch{channels[0]:04}' not in fl.name:
            break
    n_time_chunks = i + 1


    data_files_per_freq = {}
    for channel in channels:
        data_files_per_freq[channel] = []
        for tch in range(n_time_chunks):
            this = data_path / f'{args.sky_model}_fch{channel:04}_nt17280_chunk{tch}.uvh5'
            assert this.exists()
            data_files_per_freq[channel].append(this)
    
    with Pool(args.ncores) as p:
        p.map(concat_blt, channels)
