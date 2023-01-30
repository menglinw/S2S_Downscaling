#!/bin/bash
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --account=mereditf_284


module load gcc
module load intel
module load cuda
module load conda
eval "$(conda shell.bash hook)"
conda activate cGAN_space

python3 /scratch1/menglinw/S2S_Downscaling/large_model/get_indata_maps.py /scratch1/menglinw/Results/1_8_1

