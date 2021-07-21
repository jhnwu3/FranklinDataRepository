#!/bin/bash
#SBATCH --time=0-40:10:00 
#SBATCH --job-name=PSO_ODE_Nonlinear
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/NLF%j.txt
export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE
set -e
echo "PSO_S_ODE_NonLinear"
./PSO_S_ODE_NLF
mv GBMAT.csv GBMAT_$(date +%F-%H%M).csv