#!/bin/bash
save_path=/scratch1/menglinw/S2S_temp_data
cur_path=`pwd`

for season in 1 2 3 4
do
  mkdir $save_path/Season$season
  for area in 1 2 3 4
  do
    mkdir $save_path/Season$season/Area$area
    cp batch_run.sh $save_path/$season/$area
    cd $save_path/$season/$area
    echo "python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py $save_path/Season$season/Area$area $season $area">>batch_run.sh
    echo "python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py $save_path/Season$season/Area$area $season $area"
    sbatch batch_run.sh
    cd cur_path
  done
done