from pathlib import Path
import argparse
import yaml
import numpy as np

parser = argparse.ArgumentParser(description="Check whether all simulations have been run")

parser.add_argument("sky_model", type=str, help='the sky model to check')
parser.add_argument('--ntimes', type=int, default=17280)
parser.add_argument("--chunks", type=int, default=3)
parser.add_argument("--freq-range", type=int, nargs=2, default=[0,1536])

args = parser.parse_args()

REPODIR = Path(__file__).parent.absolute()

pth = Path(f"outputs/{args.sky_model}/nt{args.ntimes}")

if not pth.exists():
    raise ValueError(f"path {pth} does not exist")

files = pth.glob("*.uvh5")
fnames = [f.name for f in files]

with open(REPODIR/'freqs.yaml', 'r') as fl:
    freq_info = yaml.load(fl, Loader=yaml.FullLoader)

FREQ_ARRAY = np.arange(freq_info['start'], freq_info['end'] + freq_info['delta']/2, freq_info['delta'])

n = len(FREQ_ARRAY)
assert args.freq_range[0] >= 0
assert args.freq_range[1] <= n

for i in range(args.freq_range[0], args.freq_range[1]):
    for j in range(args.chunks):
        wrong_file = REPODIR / f'outputs/{args.sky_model}__fch{i:04}_nt{args.ntimes}_chunk{j}.uvh5'
        right_file = pth / f'{args.sky_model}_fch{i:04}_nt{args.ntimes}_chunk{j}.uvh5'
        
        if wrong_file.exists():
            wrong_file.rename(right_file)
            
        if right_file.name not in fnames:
            print(f"Missing {right_file.name}")
        
