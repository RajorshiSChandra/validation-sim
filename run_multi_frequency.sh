#!/bin/bash

skymodel=$1
minfreq=$2
maxfreq=$3

for ((freq=$minfreq ; freq<=$maxfreq ; freq++ ))
do
    ./run_single_frequency.sh $skymodel $freq "${@:4}"
done
