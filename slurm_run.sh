#!/bin/bash
#SBATCH --time=0-00:10:00 
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
gnuplot plot_species.p
gnuplot plot_cost.p
gnuplot plot_cost_rand.p
mv nonlinearODE.png outputs
mv costs_dist.png outputs
mv costs_dist_rand.png outputs
mv Protein_Cost_Dist.txt outputs
mv Protein_Cost_Dist_Rand.txt outputs
mv Protein_Cost_Labeled.txt outputs