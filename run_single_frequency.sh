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
trace="time"
chunks=1

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
	--trace)
	    loglevel="MEMTRACE";
	    trace="PYTHONTRACEMALLOC=1 time";
	    ;;
        --chunks)
	    chunks=${2}; shift;
	    ;;
    esac
    shift
done

echo "Running vis-cpu for the ${model} model in channel $freq with time-split $chunks"

printf -v padded_freq "%04d" $freq

if [[ ${profile} == 1 ]]
then
    if [[ ${gpu} == 1 ]]
    then
	trace="nvprof"
    fi
    
    profile="--profile profiling/${model}-fch${padded_freq}-gpu${gpu}-nt${nt}.profile.txt"
    echo "Running with profiling, with ntimes=${nt}"
fi


gpu_sbatch=""
module=""
if [ ${gpu} == 1 ]
then
    echo "Running on GPU"
    if [[ "$(hostname)" == *"bridges2"* ]]
    then
	partition="GPU-shared"
	module="module load cuda"
    else
	partition=GPU
    fi
    gpu_sbatch="#SBATCH --gpus=1"
    settings="visgpu.yaml"
else
    echo "Running on CPU"
    partition="Main"
    if [[ "$(hostname)" == *"agave"* ]]
    then
	partition="serial"
    fi
    settings="viscpu.yaml"
fi

for (( ch=0; ch<$chunks; ch++ ))
do
    obsparams="config_files/obsparams/${model}/nt${nt}_spl${chunks}/fch${padded_freq}_chunk${ch}.yaml"
    if [[ ! -f "$obsparams" ]]
    then
	echo "No file ${obsparams}!"
	exit
    fi
done

runtime="0-00:30:00"
if [ "${dry}" != "--dry-run" ] && [ ${gpu} != 1 ]
then
    if [[ $nt -gt 200 ]]
    then
	runtime="0-02:00:00"
    fi

    if [[ $nt -gt 7200 ]]
    then
	runtime="0-30:00:00"
    fi
fi
echo "GOING TO RUN ON ${partition} PARTITION FOR ${runtime}"

sbatch <<EOT
#!/bin/bash
#SBATCH -o logs/vis/${model}/%j.out
#SBATCH --job-name=${model}${freq}
#SBATCH --mem=23GB
#SBATCH --partition=${partition}
#SBATCH -c 1
#SBATCH -t ${runtime}
#SBATCH -N 1
${gpu_sbatch}

lscpu

${module}
source ~/miniconda3/bin/activate
conda activate h4c
echo $(which python)

for (( ch=0; ch<$chunks; ch++ ))
do
   echo "Running Time-Chunk ${ch}"
   obsparams="config_files/obsparams/${model}/nt${nt}_spl${chunks}/fch${padded_freq}_chunk${ch}.yaml"	
   ${trace} hera-sim-vis.py ${obsparams} ${settings} --compress --normalize_beams --fix_autos ${profile} --log-level ${loglevel} ${dry}   
done

EOT


