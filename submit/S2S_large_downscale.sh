#!/bin/bash
save_path=/scratch1/menglinw/Results/1_8_1
cur_path=`pwd`

# run data processing script
cp init.sh $save_path/
cd $save_path
echo "python3 /scratch1/menglinw/S2S_Downscaling/large_model/get_large_data_ready.py $save_path">>init.sh
jid1=$(sbatch --parsable --wait init.sh)
cd $cur_path

wait
# start training model
train_jids=""
for season in 1 2 3 4
do
  for area in 0 1
  do
    cp batch_run_large.sh $save_path/Season$season/Area$area/train_model.sh
    cd $save_path/Season$season/Area$area
    echo "python3 /scratch1/menglinw/S2S_Downscaling/large_model/train_large_model.py $save_path/Season$season/Area$area">>train_model.sh
    train_jids="$train_jids:$(sbatch --dependency=afterok:$jid1 --parsable train_model.sh)"
    echo "Submited Model Training: $save_path/Season$season/Area$area"
    cd $cur_path
  done
done


# downscale
echo $train_jids
cp batch_run_large.sh $save_path/evaluate.sh
cd $save_path
echo "python3 /scratch1/menglinw/S2S_Downscaling/large_model/get_large_downscale_ready.py $save_path">>evaluate.sh
sbatch --dependency=afterok$train_jids evaluate.sh
echo "Submited Evaluation"

#sbatch --dependency=afterok$train_jids YY.sh