#!/bin/bash

model=${1}
ntimes=17280

if [ ! -z "$2" ]
  then
    ntimes=${2}
fi

echo "CREATING MODEL $model with $ntimes times"

partition="Main"
if [[ "$(hostname)" == *"agave"* ]]
then
    partition="htc"
fi

sbatch <<EOT
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -o logs/makeparams/%j.out
#SBATCH -p ${partition}
#SBATCH -t 00:30
#SBATCH -N 1

cd config_files

source ~/miniconda3/bin/activate
conda activate h4c
echo $(which python)

python make_obsparams_per_freq.py ${model} --ntimes ${ntimes}

cd ..
EOT
