import argparse
from hera_sim.visibilities import ModelData
from line_profiler import LineProfiler
from pyuvdata import uvdata
from pyuvdata.utils import get_baseline_redundancies
from pathlib import Path
import numpy as np

def do_the_stuff():
    data_model = ModelData.from_config(
        args.obsparam, normalize_beams=True
    )

    print("Writing Full Output... ")
    data_model.uvdata.write_uvh5('outputs/empty_full_file.uvh5', clobber=True, run_check_acceptability=False, run_check=False)
    print("Done Writing.")

    print("Compressing...")
    if not Path('tmp-compress.npy').exists():
        red_gps = data_model.uvdata.get_redundancies(
            tol=1.0, include_conjugates=True
        )[0]
        bl_ants = [data_model.uvdata.baseline_to_antnums(gp[0]) for gp in red_gps]
        blt_inds = data_model.uvdata._select_preprocess(
            antenna_nums=None,
            antenna_names=None,
            ant_str=None,
            bls=bl_ants,
            frequencies=None,
            freq_chans=None,
            times=None,
            time_range=None,
            lsts=None,
            lst_range=None,
            polarizations=None,
            blt_inds=None,
        )[0]
        
        np.save('tmp-compress.npy', blt_inds)
    else:
        blt_inds = np.load('tmp-compress.npy')
                
    data_model.uvdata._select_by_index(
        blt_inds, None, None, "Compressed by redundancy", keep_all_metadata=True
    )
    #data_model.uvdata.compress_by_redundancy(keep_all_metadata=True)
    print("Done Compressing...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vis_cpu via hera_sim given an obsparam."
    )
    parser.add_argument("obsparam", type=str, help="pyuvsim-formatted obsparam file.")
    args = parser.parse_args()

    profiler = LineProfiler()
    profiler.add_function(uvdata.UVData.compress_by_redundancy)
    profiler.add_function(uvdata.UVData.get_redundancies)
    profiler.add_function(get_baseline_redundancies)
    profiler.add_function(uvdata.UVData.write_uvh5)
    profiler.add_function(uvdata.UVData.select)
    profiler.add_function(uvdata.UVData._select_preprocess)

    profiler.runcall(do_the_stuff)


    profiler.print_stats()
    
