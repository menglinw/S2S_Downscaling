#!/bin/bash
save_path=/scratch1/menglinw/test
cur_path=`pwd`

# run data processing script
cp batch_run.sh $save_path/init.sh
echo "python3 /scratch1/menglinw/S2S_Downscaling/get_data_ready.py $save_path">>init.sh
jid1=$(sbatch --parsable --wait init.sh)
cd $cur_path

wait
# start training model
train_jids=""
for season in 1 2 3 4
do
  for area in {0..9}
  do
    cp batch_run.sh $save_path/Season$season/Area$area/train_model.sh
    cd $save_path/Season$season/Area$area
    echo "python3 /scratch1/menglinw/S2S_Downscaling/train_model.py $save_path/Season$season/Area$area">>train_model.sh
    train_jids="$train_jids:$(sbatch --dependency=afterok:$jid1 train_model.sh)"
    echo "Submited Model Training: $save_path/Season$season/Area$area"
    cd $cur_path
  done
done


# downscale
echo $train_jids
#sbatch --dependency=afterok$train_jids YY.sh