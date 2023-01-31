#!/bin/bash

model=${1}
channel=${2}

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
#SBATCH --mem=4000MB
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${email}

lscpu

source ~/miniconda3/bin/activate
conda activate h4c

echo "PYTHON ENV: $(which python)"
time python combine_time_chunk.py ${model} --channels ${channel}

EOT
