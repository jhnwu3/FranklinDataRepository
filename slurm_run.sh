#!/bin/bash
#SBATCH --time=0-00:01:00 
#SBATCH --job-name=test
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=slurm.out
#SBATCH --ntasks-per-node=5
set -e
echo "This is a test run by John Wu"
export OMP_NUM_THREADS=5
./PSO
mv First_Particle.txt outputs
mv Nonlinear6_Cov_Corr_Mats.txt outputs
mv Nonlinear6_Cov_Corr_Mats_t2.txt outputs
mv Nonlinear6_Cov_Corr_Mats_t3.txt outputs
mv nonlinearODE.png outputs