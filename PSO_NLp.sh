#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=PSO_ODE_Nonlinear
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/NLP%j.txt
#SBATCH --partition=general
#SBATCH --ntasks-per-node=30
set -e
echo "PSO_S_ODE_NonLinear"
./PSO_NL
mv GBMAT.csv GBMAT_$(date +%F-%H%M).csv  