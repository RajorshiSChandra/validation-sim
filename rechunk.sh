#!/bin/bash

#SBATCH --job-name=rechunk
#SBATCH --output=/lustre/aoc/projects/hera/Validation/H1C_IDR3/logs/rechunk_%A_%a.out
#SBATCH --partition=hera
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-0
#SBATCH --mem=15G
#SBATCH --array=0-95%16

source ~/.bashrc
conda activate hera

BASE_DIR='/ilifu/astro/projects/hera/Validation/H4C'
SKY_CMP='diffuse'
DATA_DIR="$BASE_DIR/raw_data/$SKY_CMP"
SAVE_DIR="$BASE_DIR/chunked_data/$SKY_CMP"
SCRIPT_DIR="$BASE_DIR/scripts"
READ_GLOB='*_fch????.uvh5'  # Change this to change num of freqs.
# ARRAY_LAYOUT="$BASE_DIR/antenna_select/array_layout.csv"
ARRAY_LAYOUT=""

# echo "Running on node ${SLURMD_NODENAME}"
# other_jobs=$(squeue -w ${SLURMD_NODENAME})
# echo "Other jobs running on this node:"
# echo $(squeue -w ${SLURMD_NODENAME})
echo $(date)

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate h4c
echo $(which python)

cd $SCRIPT_DIR
echo python rechunk.py $DATA_DIR $SAVE_DIR -r $READ_GLOB -s $SKY_CMP \
            --n_stride $SLURM_ARRAY_TASK_ID --n_times_to_load 180 \
            --array_layout $ARRAY_LAYOUT --clobber --compress
python rechunk.py $DATA_DIR $SAVE_DIR -r $READ_GLOB -s $SKY_CMP \
       --n_stride $SLURM_ARRAY_TASK_ID --n_times_to_load 180 \
       --array_layout $ARRAY_LAYOUT --clobber --compress


echo $(date)
