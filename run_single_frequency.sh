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
clobber=0
mem=23
cpus_per_task=1
runtime=""

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
	--clobber)
	    clobber=${2}; shift;
	    ;;
    --mem)
        mem=${2}; shift;
        ;;
    --cpus-per-task)
        cpus_per_task=${2}; shift;
        ;;
    --runtime)
       runtime=${2}; shift;
        ;;
    esac
    shift
done

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
    module="module load cuda/10.2.0"
    if [[ "$(hostname)" == *"bridges2"* ]]
    then
        partition="GPU-shared"
        gpu_sbatch="#SBATCH --gpus=1"
    else
        partition=GPU
        gpu_sbatch="#SBATCH --gres=gpu:1"
    fi
    settings="visgpu.yaml"
    code="vis-gpu"
else
    partition="Main"
    if [[ "$(hostname)" == *"agave"* ]]
    then
        partition="serial"
    fi
    settings="viscpu.yaml"
    code="vis-cpu"
fi


msg="Running $code: ${model} | chan=${padded_freq} | chunks=$chunks."

run=0
for (( ch=0; ch<$chunks; ch++ ))
do
    obs="config_files/obsparams/${model}/nt${nt}_spl${chunks}/fch${padded_freq}_chunk${ch}.yaml"
    if [[ ! -f "$obs" ]]
    then
        echo "No file ${obs}!"
        exit
    fi

    if [ -f "outputs/${model}/nt${nt}/${model}_fch${padded_freq}_nt${nt}_chunk${ch}.uvh5" ]
    then
        ((run+=1))
    fi
done

if [ $run == $chunks ] && [ $clobber == 0 ]
then
    echo "$msg All chunks already run. Exiting"
    exit
fi

echo $msg


if [ -z "${runtime}" ]
then
    if [ "${dry}" != "--dry-run" ]
    then
        if [ ${gpu} != 1 ]
        then	       
            if [[ $nt -gt 200 ]]
            then
                runtime="0-02:00:00"
            fi
	
            if [[ $nt -gt 7200 ]]
            then
                runtime="0-30:00:00"
            fi
        else
            if [ $nt -gt 2000 ]
            then
                runtime="0-00:25:00"
            fi
	    if [ "$model" == *"diffuse"* ]
	    then
		runtime="0-01:00:00"
	    fi
        fi
    fi
fi



echo "GOING TO RUN ON ${partition} PARTITION FOR ${runtime}"

# Make sure that the slurm log directory exists. 
# Otherwise, the job will terminate
log_dir=logs/vis/${model}
if [ ! -d ${log_dir} ]; then
  mkdir -p ${log_dir}
fi

email=$(git config user.email)

sbatch <<EOT
#!/bin/bash
#SBATCH -o ${log_dir}/%j.out
#SBATCH --job-name=${model}${freq}
#SBATCH --mem=${mem}GB
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --time=${runtime}
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${email}

${gpu_sbatch}

lscpu

${module}
source ~/miniconda3/bin/activate
conda activate h4c

echo "PYTHON ENV: $(which python)"
echo "SETTINGS: ${settings}"
echo "NCHUNKS: ${chunks}"
echo "PROFILE: ${profile}"
echo "LOGLEVEL: ${loglevel}"
echo "DRY: ${dry}"
echo "NTIMES: ${nt}"
echo "FREQ: ${padded_freq}"
echo "TRACE: ${trace}"

for ((c=0 ; c<$chunks ; c++))
do
   outfile="outputs/${model}/nt${nt}/${model}_fch${padded_freq}_nt${nt}_chunk\$c.uvh5"

   if [ -f \$outfile ] && [ ${clobber} == 0 ]
   then
	continue
   fi

   echo "Running Time-Chunk \$c"
   obsparams="config_files/obsparams/${model}/nt${nt}_spl${chunks}/fch${padded_freq}_chunk\${c}.yaml"	
   command="${trace} hera-sim-vis.py \$obsparams ${settings} --compress outputs/compression-cache/nt${nt}_chunk${chunks}.npy --normalize_beams --fix_autos ${profile} --log-level ${loglevel} ${dry}"
   echo "OBSPARAMS: \${obsparams}"
   echo "RUNNING: \$command"
   eval \$command
done
EOT
