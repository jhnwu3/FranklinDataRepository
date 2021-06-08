#!/bin/bash
#SBATCH --time=0-00:01:00 
#SBATCH --job-name=test
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=slurm.out
#SBATCH --ntasks-per-node=20
set -e
echo "Testing Parallel Cost Functions"
export OMP_NUM_THREADS=20
./PSO
mv First_Particle.txt outputs
mv Nonlinear6_Cov_Corr_Mats.txt outputs
mv Nonlinear6_Cov_Corr_Mats_t2.txt outputs
mv Nonlinear6_Cov_Corr_Mats_t3.txt outputs
gnuplot plot.p
mv nonlinearODE.png outputs