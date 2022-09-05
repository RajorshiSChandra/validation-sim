#!/bin/bash

#SBATCH -c 1
#SBATCH -t 0-01:00:00
#SBATCH -o logs/empty-data/%J.out
#SBATCH --mem 32GB

source activate ~/miniconda3/bin/activate
conda activate h4c

time python create_empty_data.py config_files/obsparams/ptsrc/nt17280_spl3/fch0100_chunk1.yaml
