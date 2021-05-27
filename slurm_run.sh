#!/bin/bash
#SBATCH --time=1-00:01:00 
#SBATCH --job-name=test
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20 
#SBATCH --output=slurm.out
set -e
echo "This is a test run by John Wu"
export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE
./ODE 
