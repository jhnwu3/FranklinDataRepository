#!/bin/bash
#SBATCH --time=0-50:10:00 
#SBATCH --job-name=contour
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/contour%j.txt
#SBATCH --cpus-per-task=1
# load all modules, build terminal code, move all outputs into output folders.

./contour

mv eqwts_contour.csv contour_data/eq$(date +%F-%H%M%s).csv  
mv uneqwts_contour.csv contour_data/uneq$(date +%F-%H%M%s).csv  