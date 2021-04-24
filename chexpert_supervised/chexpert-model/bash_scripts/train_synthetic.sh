#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=60:00:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:4

#SBATCH --job-name="train_synthetic"
#SBATCH --output=train_synthetic-%j.out

SAVE_DIR='/deep/group/chexperturbed/runs/2019-04-25-00.37.28.808433__minhphu'
TRAIN_CSV='/deep/group/chexperturbed/data/CheXpert/synthetic_final/random/level_5/train_with_normal.csv'
VALID_CSV='/deep/group/chexperturbed/data/CheXpert-original/prosp500_all.csv'
# Ignore
echo "Running Uignore..."
IGNORE_NAME='Uone'
python ../train.py --dataset custom --train_custom_csv $TRAIN_CSV --val_custom_csv $VALID_CSV --save_dir $SAVE_DIR --batch_size 48 --iters_per_print 48 --iters_per_visual 48000 --iters_per_eval=4800 --iters_per_save=4800 --gpu_ids 0,1,2 --experiment_name ${IGNORE_NAME}_1 --num_epochs=3 --metric_name chexpert-competition-AUROC --maximize_metric True --scale 320 --max_ckpts 10 --keep_topk True && \
python ../train.py --dataset custom --train_custom_csv $TRAIN_CSV --val_custom_csv $VALID_CSV --save_dir $SAVE_DIR --batch_size 48 --iters_per_print 48 --iters_per_visual 48000 --iters_per_eval=4800 --iters_per_save=4800 --gpu_ids 0,1,2 --experiment_name ${IGNORE_NAME}_2 --num_epochs=3 --metric_name chexpert-competition-AUROC --maximize_metric True --scale 320 --max_ckpts 10 --keep_topk True && \
python ../train.py --dataset custom --train_custom_csv $TRAIN_CSV --val_custom_csv $VALID_CSV --save_dir $SAVE_DIR --batch_size 48 --iters_per_print 48 --iters_per_visual 48000 --iters_per_eval=4800 --iters_per_save=4800 --gpu_ids 0,1,2,3 --experiment_name ${IGNORE_NAME}_3 --num_epochs=3 --metric_name chexpert-competition-AUROC --maximize_metric True --scale 320 --max_ckpts 10 --keep_topk True


# Uone
# TODO

# Uzero
# TODO

# Self-train
# TODO

# 3class
# TODO

echo "Done!"
