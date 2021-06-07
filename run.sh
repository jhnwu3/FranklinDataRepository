# load all modules, build terminal code, move all outputs into output folders.
source load.sh
git pull
make
./PSO
mv Moment.csv outputs
mv First_Particle.txt outputs
mv Nonlinear6_Cov_Corr_Mats.txt outputs
mv ODE_Const_Soln.csv outputs
gnuplot plot.p
mv nonlinearODE.png outputs
