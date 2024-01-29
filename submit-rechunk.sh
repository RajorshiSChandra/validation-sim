#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --mem=23GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

source ~/miniconda3/bin/activate
conda activate h4c-sim

time python core/rechunk-fast.py \
  --r-prototype "eor_fch{channel:04d}_nt17280_chunk0.uvh5" \
  --chunk-size 2 \
  --channels 0~1536 \
  --sky-cmp eor \
  --assume-same-blt-layout \
  --is-rectangular \
  --nthreads 16 \
  --conjugate \
  --remove-cross-pols \
  /home/herastore02-1/Validation/H4C-Simulations/eor/nt17280 \
  outputs/eor/rechunk
