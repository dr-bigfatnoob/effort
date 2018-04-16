#!/bin/bash

#SBATCH --job-name effort
#SBATCH -N 8
#SBATCH -p opteron
#SBATCH -x c[79-98,101-107]
# Use modules to set the software environment

python runner.py