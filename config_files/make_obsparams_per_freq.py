from copy import deepcopy
import yaml
import numpy as np
from pathlib import Path
import argparse

import argparse

parser = argparse.ArgumentParser(description='Create a set of obsparams for a given sky model.')

parser.add_argument('sky_model', type=str,
                    help='the sky model for which to create obsparams')

parser.add_argument("--decimate", type=int, help="decimate the times by this amount", default=1)

args = parser.parse_args()

THISDIR = Path(__file__).parent.absolute()
REPODIR = THISDIR.parent

with open(REPODIR/'freqs.yaml', 'r') as fl:
    freq_info = yaml.load(fl, Loader=yaml.FullLoader)
    
# Frequencies to simulate -- this is set by
# UVData files that Aaron EW provided.
FREQ_ARRAY = np.arange(
    freq_info['start'],
    freq_info['end'],
    freq_info['delta'],
)


# Define a custom yaml representer for strings
def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

yaml.add_representer(str, quoted_presenter)


# Default config for templatingfrom jinja2 import Template
cfg_default = {
    'filing': {
        'outdir': f'{REPODIR}/outputs',
        'outfile_name': "{sky_model}_{freq}",
        'output_format': 'uvh5',
        'clobber': True
    },
    'freq': {
        'Nfreqs': 1, 
        'channel_width': 97656.25, 
        'start_freq': 100000000.0
    },
    'sources': {
        'catalog': f'{REPODIR}/sky_models/{args.sky_model}/<FREQ>.skyh5'
    },
    'telescope': {
        'array_layout': f'{THISDIR}/H4C-antennas.csv',
        'telescope_config_name': f'{THISDIR}/h4c_idr2.1_teleconfig.yaml'
    },
    'time': {
        'Ntimes': 17280 // args.decimate,
        'integration_time': 4.986347833333333 * args.decimate,
        'start_time': 2458208.916228965
    }
}

obsp_dir = THISDIR / 'obsparams' / args.sky_model

if not obsp_dir.exists():
    obsp_dir.mkdir(parents=True)
    
for i, f in enumerate(FREQ_ARRAY):
    outfile_name = cfg_default['filing']['outfile_name'].format(sky_model=args.sky_model, freq=f'_fch{i:04d}')
    catalog = cfg_default['sources']['catalog'].replace('<FREQ>', f'fch{i:04d}')
    cfg_f = deepcopy(cfg_default)
    cfg_f['filing']['outfile_name'] = outfile_name
    cfg_f['sources']['catalog'] = catalog
    cfg_f['freq']['Nfreqs'] = 1
    cfg_f['freq']['start_freq'] = float(f)
    
    obsparams_file = f'{obsp_dir}/fch{i:04d}.yaml'
    
    with open(obsparams_file, 'w', ) as stream:
        yaml.dump(cfg_f, stream, default_flow_style=False, sort_keys=False)
