#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=contour
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/contour%j.txt
#SBATCH --cpus-per-task=1
# load all modules, build terminal code, move all outputs into output folders.

./contour

mv GBMATP.csv data/GBMAT_para_$(date +%F-%H%M%s).csv  
mv runs.csv data/run$(date +%F-%H%M%s).csv  