#!/bin/bash
#SBATCH --time=0-10:10:00 
#SBATCH --job-name=PSO_ODE_Nonlinear
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/nonlinear_out%j.txt
#SBATCH --ntasks-per-node=8
# load all modules, build terminal code, move all outputs into output folders.
source load.sh
git pull
make
./para
