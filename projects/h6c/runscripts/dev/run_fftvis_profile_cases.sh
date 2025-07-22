
common="--log-level INFO --skip-existing --do-time-chunks 0 --channels 1 --profile --profile-timer-unit 1e-2 --slurm-override time '00:20:00' --simulator fftvis --rerun-existing --n-time-chunks 1080 --spline-interp-order 1 --redundant"

#./vsim.py runsim ${common} --sky-model gsm_nside128 --layout FULL
#./vsim.py runsim ${common} --sky-model gsm_nside128 --layout HEX

#./vsim.py runsim ${common} --sky-model gsm_nside256 --layout FULL
#./vsim.py runsim ${common} --sky-model gsm_nside256 --layout HEX

#./vsim.py runsim ${common} --sky-model gsm_nside512 --layout FULL
#./vsim.py runsim ${common} --sky-model gsm_nside512 --layout HEX

#./vsim.py runsim ${common} --sky-model gsm_nside1024 --layout FULL
#./vsim.py runsim ${common} --sky-model gsm_nside1024 --layout HEX


#./vsim.py runsim ${common} --sky-model gsm_nside1024 --layout HEX --n-time-chunks 540

# Now also run tests of scaling with cpu cores
common="${common} --sky-model gsm_nside1024 --layout FULL --slurm-override partition 'RM-shared'"
#./vsim.py runsim ${common} --prefix ncores1 --slurm-override ntasks 1
#./vsim.py runsim ${common} --prefix ncores2 --slurm-override ntasks 2  // these get OOM'd
./vsim.py runsim ${common} --prefix ncores4 --slurm-override ntasks 4
#./vsim.py runsim ${common} --prefix ncores8 --slurm-override ntasks 8
#./vsim.py runsim ${common} --prefix ncores16 --slurm-override ntasks 16
#./vsim.py runsim ${common} --prefix ncores32 --slurm-override ntasks 32
#./vsim.py runsim ${common} --prefix ncores64 --slurm-override ntasks 64



