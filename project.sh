#!/bin/bash
#$ -N project
#$ -q class8-intel 
#$ -pe openmp 8

# Module load gcc compiler version 4.9.2
module load  gcc/4.9.2

# Runs a bunch of standard command-line
# utilities, just as an example:

./NCOpenMP