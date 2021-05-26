#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=general
#SBATCH --cpus-per-task=2 
set -e
echo "This is a test run by John Wu"
./ODE
sleep 30