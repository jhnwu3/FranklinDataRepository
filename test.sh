#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=ODE_Plotting
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/test%j.txt
#SBATCH --ntasks-per-node=4
# load all modules, build terminal code, move all outputs into output folders.
source load.sh
git pull
make
./para
