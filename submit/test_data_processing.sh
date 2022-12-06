#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --account=mereditf_284


module load gcc
module load intel
module load cuda
module load conda
eval "$(conda shell.bash hook)"
conda activate cGAN_space

python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py /scratch1/menglinw/test