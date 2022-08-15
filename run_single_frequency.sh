#!/bin/bash

model=${1}
freq=${2}
shift
shift


gpu=0
profile=""
nt=17280

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
    esac
    shift
done

echo "Running vis-cpu for the ${model} model in channel ${2}"

printf -v padded_freq "%04d" $freq


if [ ${profile} == 1 ]
then
    profile="--profile profiling/${model}-fch${padded_freq}-gpu${gpu}-nt${nt}.profile.txt -p pyuvdata.uvbeam:UVBeam.interp"
    echo "Running with profiling, with ntimes=${nt}"
fi



if [ ${gpu} == 1 ]
then
    echo "Running on GPU"
    partition=GPU
    settings="visgpu.yaml"
else
    echo "Running on CPU"
    partition=Main
    settings="viscpu.yaml"
fi

sbatch <<EOT
#!/bin/bash
#SBATCH -o logs/vis/${model}/%j.out
#SBATCH --job-name=${model}${freq}
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=${partition}

conda activate h4c
echo $(which python)

time hera-sim-vis.py config_files/obsparams/${model}/fch${padded_freq}.yaml ${settings} --compress --normalize_beams --fix_autos ${profile}
EOT


