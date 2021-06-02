# load all modules, build terminal code, move all outputs into output folders.

source load.sh
git pull
make
./PSO
mv Moment.csv outputs
mv First_Particle.txt outputs