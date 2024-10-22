./vsim.py runsim \
  --log-level INFO --skip-existing --simulator fftvis64 \
  --redundant --layout FULL \
  --slurm-override time '03:00:00' --slurm-override partition RM  \
  --sky-model ptsrc1024 --spline-interp-order 1 \
  --prefix final \
  --n-time-chunks 1 --do-time-chunks 0 --channels 100~1200

./vsim.py runsim \
  --log-level INFO --skip-existing --simulator fftvis32 \
  --redundant --layout FULL \
  --slurm-override time '05:00:00' --slurm-override partition RM  \
  --sky-model ptsrc1024 --spline-interp-order 1 \
  --prefix final \
  --n-time-chunks 1 --do-time-chunks 0 --channels 1200~1536
