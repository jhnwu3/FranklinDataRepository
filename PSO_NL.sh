#!/bin/bash
#SBATCH --time=0-10:10:00 
#SBATCH --job-name=PSO_ODE_Nonlinear
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/NLF%j.txt
#SBATCH --partition=general
#SBATCH --ntasks-per-node=1
set -e
echo "PSO_S_ODE_NonLinear"
./PSO_S_ODE_NLF
mv GBMAT.csv GBMAT_$(date +%F-%H%M).csv  