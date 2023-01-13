#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --account=mereditf_284


module load gcc
module load intel
module load cuda
module load conda
eval "$(conda shell.bash hook)"
conda activate cGAN_space


