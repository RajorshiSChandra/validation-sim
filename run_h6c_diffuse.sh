#./vsim.py runsim \
#  --log-level INFO --skip-existing --simulator fftvis64 --force-remake-obsparams \
#  --redundant --layout FULL \
#  --slurm-override time '02:30:00' --slurm-override partition RM  \
#  --sky-model gsm_nside1024 --spline-interp-order 1 \
#  --prefix final \
#  --n-time-chunks 1 --do-time-chunks 0 --channels 0~1300

./vsim.py runsim \
  --log-level INFO --skip-existing --simulator fftvis32 --force-remake-obsparams \
  --redundant --layout FULL \
  --slurm-override time '05:00:00' --slurm-override partition RM  \
  --sky-model gsm_nside1024 --spline-interp-order 1 \
  --prefix final \
  --n-time-chunks 1 --do-time-chunks 0 --channels 1300~1536
