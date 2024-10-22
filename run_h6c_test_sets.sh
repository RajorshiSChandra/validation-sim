
common="--log-level INFO --skip-existing --profile --profile-timer-unit 1e-2 --simulator fftvis --force-remake-obsparams --redundant --layout FULL"

# We do the two different spline interp orders with different channels or time chunks so they don't overwrite each other.

# Diffuse long-time-axis tests. 3 hours for two channels to check FRF.
# longtime=" --n-time-chunks 1080 --do-time-chunks 0 --channels 1"

longtime=" --n-time-chunks 24 --do-time-chunks 0~3 --channels 1 --channels 1500"
# ./vsim.py runsim ${common} ${longtime} --prefix order1 --slurm-override time '03:00:00' --slurm-override ntasks 3 --sky-model gsm_nside512 --spline-interp-order 1 
# ./vsim.py runsim ${common} ${longtime} --prefix order3 --slurm-override time '03:00:00' --slurm-override ntasks 3 --sky-model gsm_nside512 --spline-interp-order 3
# ./vsim.py runsim ${common} ${longtime} --prefix order1 --slurm-override time '03:00:00' --slurm-override ntasks 6 --sky-model gsm_nside1024 --spline-interp-order 1
# ./vsim.py runsim ${common} ${longtime} --prefix order3 --slurm-override time '03:00:00' --slurm-override ntasks 6 --sky-model gsm_nside1024 --spline-interp-order 3

# ./vsim.py runsim ${common} ${longtime} --prefix order1 --slurm-override time '03:00:00' --slurm-override ntasks 3 --sky-model ptsrc512 --spline-interp-order 1 
# ./vsim.py runsim ${common} ${longtime} --prefix order3 --slurm-override time '03:00:00' --slurm-override ntasks 3 --sky-model ptsrc512 --spline-interp-order 3
# ./vsim.py runsim ${common} ${longtime} --prefix order1 --slurm-override time '03:00:00' --slurm-override ntasks 6 --sky-model ptsrc1024 --spline-interp-order 1
# ./vsim.py runsim ${common} ${longtime} --prefix order3 --slurm-override time '03:00:00' --slurm-override ntasks 6 --sky-model ptsrc1024 --spline-interp-order 3


# Diffuse band tests.
band="--n-time-chunks 1080 --channels 0~150 --channels 1300~1450 --do-time-chunks 0 --do-time-chunks 500 --slurm-override time '00:30:00' --slurm-override ntasks 4"
./vsim.py runsim ${common} ${band} --prefix order1 --sky-model gsm_nside512 --spline-interp-order 1
./vsim.py runsim ${common} ${band} --prefix order3 --sky-model gsm_nside512 --spline-interp-order 3
./vsim.py runsim ${common} ${band} --prefix order1 --sky-model gsm_nside1024 --spline-interp-order 1
./vsim.py runsim ${common} ${band} --prefix order3 --sky-model gsm_nside1024 --spline-interp-order 3

# ./vsim.py runsim ${common} ${band} --prefix order1 --sky-model eor_nside512 --spline-interp-order 1
# ./vsim.py runsim ${common} ${band} --prefix order3 --sky-model eor_nside512 --spline-interp-order 3
# ./vsim.py runsim ${common} ${band} --prefix order1 --sky-model eor_nside1024 --spline-interp-order 1
# ./vsim.py runsim ${common} ${band} --prefix order3 --sky-model eor_nside1024 --spline-interp-order 3




