#!/bin/bash
#SBATCH --time=0-10:10:00 
#SBATCH --job-name=PSO_ODE_Nonlinear
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=nonlinear_out.txt
#SBATCH --ntasks-per-node=1
set -e
echo "PSO_S_ODE_NonLinear"
./PSO_S_ODE_NL