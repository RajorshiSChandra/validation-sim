#!/bin/bash

model=${1}
chunk=${2}

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
#SBATCH --job-name=rechunk-${model}-${chunk}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${email}

lscpu
source ~/miniconda3/bin/activate
conda activate h4c

echo "PYTHON ENV: $(which python)"
time python rechunk-fast.py \
   --r-prototype "${model}_fch{channel:04d}_nt17280_chunk${chunk}.uvh5" \
   --chunk-size 2 \
   --channels 0~1535 \
   --sky-cmp ${model}\
   --assume-same-blt-layout \
   --is-rectangular \
   --nthreads 16 \
   ./outputs/${model}/nt17280 \
   ./outputs/${model}/nt17280/rechunk${chunk}
EOT
