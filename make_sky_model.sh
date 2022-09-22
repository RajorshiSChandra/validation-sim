#!/bin/bash

model=${1}
ncores=1
shift

nside=""

while [ $# -ne 0 ]
do
    arg="$1"
    case "$arg" in
        --nside)
            nside=${1} ${2};
            shift
            ;;
	--ncores)
	    ncores=${2};
	    shift
	    ;;
    esac
    shift
done

partition="Main"
if [[ "$(hostname)" == *"agave2"* ]]
then
    partition="htc"
fi

if [[ "$(hostname)" == *"bridges"* ]]
then
    partition="GPU-shared"
    gpu="#SBATCH --gpus=1"
fi



echo "CREATING MODEL: $model"

sbatch <<EOT
#!/bin/bash
#SBATCH -o logs/skymodel/%j.out
#SBATCH -c ${ncores}
#SBATCH -p ${partition}
#SBATCH -t 00:30
#SBATCH -N 1
${gpu}

cd sky_models

source ~/miniconda3/bin/activate
conda activate h4c
echo $(which python)

python make_${model}_model.py ${nside}

cd ..
EOT
