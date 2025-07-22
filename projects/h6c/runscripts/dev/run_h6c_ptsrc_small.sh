./vsim.py runsim \
  --log-level INFO --skip-existing --simulator fftvis --force-remake-obsparams \
  --redundant --layout FULL \
  --slurm-override time '01:00:00' --slurm-override partition RM-shared --slurm-override ntasks 8  \
  --sky-model ptsrc1024 --spline-interp-order 1 \
  --n-time-chunks 720 --do-time-chunks 0 --channels 0 --phase-center-name zenith
