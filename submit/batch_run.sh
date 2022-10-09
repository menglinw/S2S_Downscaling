#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --account=mereditf_284


module load gcc
module load intel
module load cuda
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate cGAN_space


