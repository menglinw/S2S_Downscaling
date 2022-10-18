#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --account=mereditf_284


module load gcc
module load intel
module load cuda
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate cGAN_space

python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py /scratch1/menglinw/Area1 2 13 AFG
python3 /scratch1/menglinw/S2S_Downscaling/train_model.py /scratch1/menglinw/Area1
python3 /scratch1/menglinw/S2S_Downscaling/get_downscale_ready.py /scratch1/menglinw/Area1

