#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=parallelPSO
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/test%j.txt
#SBATCH --ntasks-per-node=20
# load all modules, build terminal code, move all outputs into output folders.

./para
