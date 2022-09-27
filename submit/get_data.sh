#!/bin/bash
save_path=/scratch1/menglinw/S2S_temp_data
cur_path=`pwd`

for season in 1 2 3 4
do
  mkdir $save_path/$season
  for area in 1 2 3 4
  do
    mkdir $save_path/$season/$area
    cd $save_path/$season/$area
    echo "python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py $save_path/Season$season/Area$area $season $area"
    cd cur_path
  done
done