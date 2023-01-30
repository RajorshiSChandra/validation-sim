from pathlib import Path
import argparse
from multiprocessing import Pool

import numpy as np
from pyuvdata import UVData

VIS_DIR = Path(__file__).parent.absolute() / 'outputs'


def concat_blt(i):
    files_to_load = data_files_per_freq[i]
    outstem = data_files_per_freq[i][0].stem.replace('_chunk0', '')
    outfile = out_path / f'{outstem}.uvh5'
    
    print(f'Combining\n{files_to_load}\nWriting to {outfile}\n')
    
    uvd = UVData()
    uvd.read(files_to_load, axis='blt')
    uvd.write_uvh5(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine time-chunked visibility into single file'
    )
    parser.add_argument('sky_model', type=str)
    parser.add_argument('--chunk_size', type=int, default=3)
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

    # Convert data_files into a nested list per frequency
    nfiles = len(data_files)
    if (nfiles % args.chunk_size) == 0:
        data_files_per_freq = [data_files[i:i+args.chunk_size] 
                               for i in range(0, nfiles, args.chunk_size)]
    else:
        raise ValueError(
            "Number of files do not work with given chunk siz."
        )

    nfreqs = len(data_files_per_freq)
    
    with Pool(args.ncores) as p:
        p.map(concat_blt, range(nfreqs))