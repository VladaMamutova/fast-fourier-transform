#!/bin/bash

#SBATCH --job-name="FAST_FOURIER_TRANSFORM"
#SBATCH --partition=debug
#SBATCH --nodes=10
#SBATCH --time=0-00:10:00
#SBATCH --ntasks-per-node=1

#SBATCH --mem=1992

mpirun -np $1 ./parallel_fast_fourier_transform $2
