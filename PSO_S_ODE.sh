#!/bin/bash
#SBATCH --time=0-01:10:00 
#SBATCH --job-name=PSO_ODE
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=slurm.out
#SBATCH --ntasks-per-node=1
set -e
echo "PSO_S_ODE"
./PSO_S_ODE