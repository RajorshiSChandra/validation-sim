#!/bin/bash

model=${1}
decimate=1

if [ ! -z "$2" ]
  then
    decimate=${2}
fi

echo "CREATING MODEL: $model"

sbatch <<EOT
#!/bin/bash
#SBATCH -o logs/makeparams/%j.out

cd config_files

conda activate hera
echo $(which python)

# build_makeflow_from_config.py is in hera_opm
python make_obsparams_per_freq.py ${model} --decimate ${decimate}

cd ..
EOT
