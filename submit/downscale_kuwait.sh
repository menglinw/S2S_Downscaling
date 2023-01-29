#!/bin/bash
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --account=mereditf_284


module load gcc
module load intel
module load cuda
module load conda
eval "$(conda shell.bash hook)"
conda activate cGAN_space

python3 /scratch1/menglinw/S2S_Downscaling/large_model/downscale_kuwait.py /scratch1/menglinw/Results/1_8_1