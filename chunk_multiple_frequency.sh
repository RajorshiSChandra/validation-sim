#!/bin/bash

skymodel=$1
minfreq=$2
maxfreq=$3

for ((freq=$minfreq ; freq<=$maxfreq ; freq++ ))
do
    ./run_combine_time_chunk.sh $skymodel $freq
done
