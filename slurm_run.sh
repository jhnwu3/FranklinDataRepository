#!/bin/bash
#SAATCH --job-name=test
#SBATCH --partition=general
#SBATCH −−cpus−per−task=1
set -e
echo "This is a test run by John Wu"
./ODE
sleep 30