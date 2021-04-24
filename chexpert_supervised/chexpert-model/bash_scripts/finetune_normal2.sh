#!/bin/bash
#SBATCH --partition=deep --qos=normal
#SBATCH --time=60:00:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:4

#SBATCH --job-name="finetune"
#SBATCH --output=finetune-%j.out

echo "Running finetune on uignore"

SAVE_DIR="/deep/group/chexperturbed/runs/2019-04-23-21.26.43.341224__minhphu"

python ../train.py --ckpt_path /deep/group/chexperturbed/runs/2019-04-18-22.17.36.095031__minhphu/DenseNet121_320_1e-04_uncertainty_ignored_top10/best.pth.tar \
                   --dataset custom \
                   --train_custom_csv /deep/group/chexperturbed/data/natural/Nokiadev10K_and_NokiaNORMALS507_noflux.csv \
                   --val_custom_csv /deep/group/chexperturbed/data/CheXpert-original/prosp500_all.csv \
                   --save_dir $SAVE_DIR \
                   --experiment_name finetune_uignore \
                   --batch_size 48 \
                   --iters_per_print 48 \
                   --iters_per_visual 48000 \
                   --iters_per_eval=4800 \
                   --iters_per_save=4800 \
                   --gpu_ids 0 \
                   --num_epochs=3 \
                   --metric_name chexpert-competition-AUROC \
                   --maximize_metric True \
                   --scale 320 \
                   --max_ckpts 10 \
                   --keep_topk True 

echo "Done!"
