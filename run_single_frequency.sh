#!/bin/bash

model=${1}
freq=${2}
shift
shift


gpu=0
profile=""
nt=17280
loglevel="WARNING"
dry=""

while [ $# -ne 0 ]
do
    arg="$1"
    case "$arg" in
        --profile)
	    profile=1;
            ;;
        --gpu)
            gpu=1;
            ;;
	--ntimes)
	    nt=${2}; shift;
	    ;;
	--log)
            loglevel=${2}; shift;
            ;;
	--dry)
	    dry="--dry-run";
	    ;;
    esac
    shift
done

echo "Running vis-cpu for the ${model} model in channel $freq"

printf -v padded_freq "%04d" $freq


if [[ ${profile} == 1 ]]
then
    profile="--profile profiling/${model}-fch${padded_freq}-gpu${gpu}-nt${nt}.profile.txt"
    echo "Running with profiling, with ntimes=${nt}"
fi



if [ ${gpu} == 1 ]
then
    echo "Running on GPU"
    partition=GPU
    settings="visgpu.yaml"
else
    echo "Running on CPU"
    partition="Main"
    if [[ "$(hostname)" == *"agave"* ]]
    then
#	partition="htc"
	partition="serial"
#	if [[ $nt -gt 7200 ]]
#	then
#	    partition="serial"
#	else
#	    partition="htc"
#	fi
    fi
    settings="viscpu.yaml"
fi


obsparams="config_files/obsparams/${model}/nt${nt}/fch${padded_freq}.yaml"
if [[ ! -f "$obsparams" ]]
then
    echo "No file ${obsparams}!"
    exit
fi

runtime="0-00:30:00"
if [[ $nt -gt 200 ]]
then
    runtime="0-02:00:00"
fi

if [[ $nt -gt 7200 ]]
then
    runtime="0-30:00:00"
fi

echo "GOING TO RUN ON ${partition} PARTITION FOR ${runtime}"

sbatch <<EOT
#!/bin/bash
#SBATCH -o logs/vis/${model}/%j.out
#SBATCH --job-name=${model}${freq}
#SBATCH --mem=32GB
#SBATCH --partition=${partition}
#SBATCH -c 1
#SBATCH -t ${runtime}
#SBATCH -N 1

lscpu

source ~/miniconda3/bin/activate
conda activate h4c
echo $(which python)

PYTHONTRACEMALLOC=1 time hera-sim-vis.py ${obsparams} ${settings} --compress --normalize_beams --fix_autos ${profile} --log-level ${loglevel} ${dry}
EOT


