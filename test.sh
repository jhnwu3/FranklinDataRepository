#!/bin/bash
#SBATCH --time=0-52:10:00 
#SBATCH --job-name=parallelPSO
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/test%j.txt
#SBATCH --cpus-per-task=30
# load all modules, build terminal code, move all outputs into output folders.

./para
mv GBMATP.csv data/GBMAT_para_$(date +%F-%H%M).csv  
mv runs.csv data/run$(date +%F-%H%M).csv  