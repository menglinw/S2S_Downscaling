#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --time=2:00:00
#SBATCH --account=mereditf_284


module load gcc/8.3.0
module load intel/19.0.4
module load cuda/10.2.89
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate cGAN_space


