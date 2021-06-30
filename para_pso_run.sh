#!/bin/bash
#SBATCH --time=0-20:10:00 
#SBATCH --job-name=test
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./slurm_outputs/para%j.txt
#SBATCH --ntasks-per-node=20
set -e
echo "Testing Parallel Cost Functions"
export OMP_NUM_THREADS=20
./PSO
