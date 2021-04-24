#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="dense121"
#SBATCH --output=exp_logs/dense121-%j.out

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

cd ../moco; python main_moco.py -a densenet121 \
            --lr 1e-4 --batch-size 16 \
            --world-size 1 --rank 0 \
            --mlp --moco-t 0.2 \
            --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
            --from-imagenet \
			--aug-setting chexpert --rotate --maintain-ratio \
            --train_data data/full_train \
            --exp-name dense121

# done
echo "Done"
