#!/bin/bash

model=${1}
njobs=${2}

echo "CREATING MODEL: $model"

sbatch <<EOT
#!/bin/bash
#SBATCH -o logs/skymodel/%j.out
#SBATCH -c ${njobs}

cd sky_models

conda activate hera
echo $(which python)

# build_makeflow_from_config.py is in hera_opm
python make_${model}_model.py

cd ..
EOT
