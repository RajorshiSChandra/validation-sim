#!/bin/python

import sys
import os

skymodel=sys.argv[1]
freqfile=sys.argv[2]

with open(freqfile, 'r') as fl:
    txt = fl.read()
    if ',' in txt:
        freqs = txt.split(',')
    else:
        freqs = txt.split("\n")

args=' '.join(sys.argv[3:])

for freq in freqs:
    os.system(f"./run_single_frequency.sh {skymodel} {freq} {args}")

