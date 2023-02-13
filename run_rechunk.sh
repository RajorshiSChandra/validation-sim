#!/bin/bash

model=${1}

# Make sure that the slurm log directory exists. 
# Otherwise, the job will terminate
log_dir=logs/chunk/${model}
if [ ! -d ${log_dir} ]; then
  mkdir -p ${log_dir}
fi

email=$(git config user.email)

sbatch <<EOT
#!/bin/bash
#SBATCH -o ${log_dir}/%j.out
#SBATCH --job-name=c-${model}
#SBATCH --mem=2000MB
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${email}

lscpu
source ~/miniconda3/bin/activate
conda activate h4c

echo "PYTHON ENV: $(which python)"
time python rechunk-fast.py outputs/${model}/nt17280 outputs/${model}/chunked/ \
   --r-prototype "${model}_fch{channel:04d}_nt17280_chunk?.uvh5" \
   --channels 0~3 \
   --sky-cmp ${model}\
   --assume-same-blt-layout \
   --log-level DEBUG \
   --blt-order time baseline
EOT
