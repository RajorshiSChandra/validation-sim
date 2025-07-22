./vsim.py runsim \
  --log-level INFO --skip-existing --simulator fftvis64 \
  --redundant --layout FULL \
  --slurm-override time '03:30:00' --slurm-override partition RM  \
  --sky-model eor-grf-1024 --spline-interp-order 1 \
  --prefix final \
  --n-time-chunks 1 --do-time-chunks 0 --channels 0~1100

./vsim.py runsim \
  --log-level INFO --skip-existing --simulator fftvis32 \
  --redundant --layout FULL \
  --slurm-override time '10:00:00' --slurm-override partition RM  \
  --sky-model eor-grf-1024 --spline-interp-order 1 \
  --prefix final \
  --n-time-chunks 1 --do-time-chunks 0 --channels 1100~1536
