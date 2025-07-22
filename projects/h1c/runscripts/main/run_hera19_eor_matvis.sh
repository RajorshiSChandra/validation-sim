#./vsim.py runsim \
#  --log-level INFO --skip-existing --simulator matvis \
#  --not-redundant --layout HERA19 \
#  --slurm-override time '00:30:00' --slurm-override partition RM-shared  \
#  --slurm-override ntasks-per-node 24 --slurm-override ntasks 24 \
#  --sky-model eor-grf-1024-monopole --spline-interp-order 1 \
#  --prefix matvis-paper \
#  --n-time-chunks 17280 --do-time-chunks 0 --channels 575~706

# Monopole-only sims
./vsim.py runsim \
  --log-level INFO --skip-existing --simulator matvis \
  --not-redundant --layout HERA19 \
  --slurm-override time '00:30:00' --slurm-override partition RM-shared  \
  --slurm-override ntasks-per-node 24 --slurm-override ntasks 24 \
  --sky-model eor-grf-1024-only-monopole --spline-interp-order 1 \
  --prefix matvis-paper \
  --n-time-chunks 17280 --do-time-chunks 0 --channels 575~706
