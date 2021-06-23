#!/bin/bash
#SBATCH --time=0-10:10:00 
#SBATCH --job-name=PSO_ODE_Linear
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=slurm_PSO_S_ODE_L.out
#SBATCH --ntasks-per-node=1
set -e
echo "PSO_S_ODE_NonLinear"
./PSO_S_ODE_L