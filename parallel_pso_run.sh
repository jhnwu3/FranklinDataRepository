#!/bin/bash
#SBATCH --time=0-10:10:00 
#SBATCH --job-name=test
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=para_pso_out.txt
#SBATCH --ntasks-per-node=20
set -e
echo "Testing Parallel Cost Functions"
export OMP_NUM_THREADS=20
./PSO
