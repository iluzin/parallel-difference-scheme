#!/usr/bin/bash

module add slurm
cd $HOME
sbatch -n $1 -o output-n$1-$2.txt -e err-n$1-$2.txt -p test impi ./a.out 1 1 1 1 $2 1024
