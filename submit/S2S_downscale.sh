#!/bin/bash
save_path=/scratch1/menglinw/Results/11_10_1
cur_path=`pwd`

for season in 1 2 3 4
do
  mkdir $save_path/Season$season
  for area in {1..9}
  do
    mkdir $save_path/Season$season/Area$area
    cp batch_run.sh $save_path/Season$season/Area$area
    cd $save_path/Season$season/Area$area
    echo "python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py $save_path/Season$season/Area$area $season $area NonAFG">>batch_run.sh
    echo "python3 /scratch1/menglinw/S2S_Downscaling/train_model.py $save_path/Season$season/Area$area">>batch_run.sh
    sbatch batch_run.sh
    echo "Submited: $save_path/Season$season/Area$area"
    cd $cur_path
  done

  mkdir $save_path/Season$season/AreaAFG
  cp batch_run.sh $save_path/Season$season/AreaAFG
  cd $save_path/Season$season/AreaAFG
  echo "python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py $save_path/Season$season/AreaAFG $season 1 AFG">>batch_run.sh
  echo "python3 /scratch1/menglinw/S2S_Downscaling/train_model.py $save_path/Season$season/AreaAFG">>batch_run.sh
  sbatch batch_run.sh
  echo "Submited: $save_path/Season$season/AreaAFG"
  cd $cur_path

done