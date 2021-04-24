#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000

# only use the following on partition with GPUs
#SBATCH --gres=gpu:4

#SBATCH --job-name="moco-v1-lincls"
#SBATCH --output=exp_logs/v1-lincls-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample job
NPROCS=`sbatch --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

cd ../moco; python main_lincls.py -a resnet50 --lr 30.0 --batch-size 256 \
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
        --world-size 1 --rank 0 \
        --train_data chexpert-v10-small-as-imagenet/data/actual_train \
        --val_data chexpert-v10-small-as-imagenet/data/actual_valid \
        --test_data chexpert-v10-small-as-imagenet/data/valid \
        --from-imagenet \
        --exp-name moco_v1_lincls
# done
echo "Done"
