#!/bin/bash
save_path=/scratch1/menglinw/S2S_temp_data
cur_path=`pwd`

for season in 1 2 3 4
do
  for area in 1 2 3 4
  do
    cp batch_run.sh $save_path/Season$season/Area$area/batch_run_model.sh
    cd $save_path/Season$season/Area$area
    echo "python3 /scratch1/menglinw/S2S_Downscaling/train_model.py $save_path/Season$season/Area$area">>batch_run_model.sh
    echo "python3 /scratch1/menglinw/S2S_Downscaling/train_model.py $save_path/Season$season/Area$area"
    sbatch batch_run_model.sh
    cd $cur_path
  done
done