# load all modules, build terminal code, move all outputs into output folders.
source load.sh
git pull
make
./PSO
mv First_Particle.txt outputs
mv Nonlinear6_Cov_Corr_Mats.txt outputs
mv Nonlinear6_Cov_Corr_Mats_t2.txt outputs
mv Nonlinear6_Cov_Corr_Mats_t3.txt outputs
gnuplot plot_species.p
gnuplot plot_cost.p
mv nonlinearODE.png outputs
mv costs_dist.png outputs
mv Protein_Cost_Dist.txt outputs
mv Protein_cost_Labeled.txt outputs