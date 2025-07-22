
common="--log-level INFO --skip-existing --gpu --do-time-chunks 0 --channels 100 --profile --slurm-override time '01:00:00'"

./vsim.py runsim ${common} --n-time-chunks 24 --sky-model ptsrc256 --layout FULL
./vsim.py runsim ${common} --n-time-chunks 24 --sky-model ptsrc256 --layout HEX
./vsim.py runsim ${common} --n-time-chunks 24 --sky-model ptsrc256 --layout H4C

./vsim.py runsim ${common} --n-time-chunks 36 --sky-model ptsrc256 --layout FULL
./vsim.py runsim ${common} --n-time-chunks 36 --sky-model ptsrc256 --layout HEX
./vsim.py runsim ${common} --n-time-chunks 36 --sky-model ptsrc256 --layout H4C

./vsim.py runsim ${common} --n-time-chunks 48 --sky-model ptsrc512 --layout FULL
./vsim.py runsim ${common} --n-time-chunks 48 --sky-model ptsrc512 --layout HEX
./vsim.py runsim ${common} --n-time-chunks 48 --sky-model ptsrc512 --layout H4C

./vsim.py runsim ${common} --n-time-chunks 60 --sky-model ptsrc512 --layout FULL
./vsim.py runsim ${common} --n-time-chunks 60 --sky-model ptsrc512 --layout HEX
./vsim.py runsim ${common} --n-time-chunks 60 --sky-model ptsrc512 --layout H4C 

./vsim.py runsim ${common} --n-time-chunks 24 --sky-model ptsrc128 --layout FULL
./vsim.py runsim ${common} --n-time-chunks 24 --sky-model ptsrc128 --layout HEX
./vsim.py runsim ${common} --n-time-chunks 24 --sky-model ptsrc128 --layout H4C

./vsim.py runsim ${common} --n-time-chunks 36 --sky-model ptsrc128 --layout FULL
./vsim.py runsim ${common} --n-time-chunks 36 --sky-model ptsrc128 --layout HEX
./vsim.py runsim ${common} --n-time-chunks 36 --sky-model ptsrc128 --layout H4C 
