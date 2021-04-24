import datetime
import os


SBATCH_SCRIPT = \
'''#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="SB_JOBNAME"
#SBATCH --output=exp_logs/SB_JOBNAME-%j.out

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

cd ../moco; python main_moco.py -a SB_MODEL \\
            --lr SB_LR --batch-size SB_BATCH_SIZE \\
            --epochs SB_EPOCHS \\
            --world-size 1 --rank 0 \\
            --mlp --moco-t 0.2 SB_FROM_IMAGENET \\
            --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \\
			--aug-setting chexpert --rotate SB_ROTATION --maintain-ratio \\
            --train_data /deep/group/data/moco/chexpert-proper-test/data/full_train \\
            --exp-name SB_EXPNAME

# done
echo "Done"
'''

BASH_SCRIPT = \
'''cd /home/jby/aihc-spring20-fewer/moco; python main_moco.py -a SB_MODEL \\
            --lr SB_LR --batch-size SB_BATCH_SIZE \\
            --world-size 1 --rank 0 \\
            --mlp --moco-t 0.2 SB_FROM_IMAGENET \\
            --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \\
			--aug-setting chexpert --rotate SB_ROTATION --maintain-ratio \\
            --train_data /home/jby/CheXpert/full_train \\
            --exp-name SB_EXPNAME 2>&1 | tee /home/jby/chexpert_experiments/jby/SB_EXPNAME_log.txt
'''


LR_SHORT  = {
                1e-7: '1n7',
                1e-6: '1n6',
                5e-5: '5n5',
                3e-5: '3n5',
                2e-5: '2n5',
                1e-5: '1n5',
                1e-4: '1n4',
                1e-3: '1n3',
                1e-2: '1n2',
                5e-2: '5n2',
                5e-4: '5n4'
            }

MODEL_SHORT_NAME_MAP = {'resnet18': 'r8',
                        'resnet50': 'r5',
                        'densenet121': 'd1'}

def gen_script(model, lr, batch_size, imagenet, epoch, gcp):

    today = datetime.datetime.now()
    strtoday = today.strftime('%Y%m%dh%H')

    sb_model = model
    sb_lr = str(lr)
    sb_epoch = str(epoch)
    sb_batch_size = str(batch_size)
    sb_from_imagenet = '--from-imagenet' if imagenet else ''
    sb_rotation = str(10)
    sb_jobname = f'{MODEL_SHORT_NAME_MAP[sb_model]}{"w" if imagenet else "o"}{LR_SHORT[lr]}{batch_size}'
    sb_expname = f'{sb_jobname}_{strtoday}'

    if not gcp:
        script = SBATCH_SCRIPT
    else:
        script = BASH_SCRIPT

    script = script.replace('SB_JOBNAME', sb_jobname)
    script = script.replace('SB_MODEL', sb_model)
    script = script.replace('SB_LR', sb_lr)
    script = script.replace('SB_EPOCH', sb_epoch)
    script = script.replace('SB_BATCH_SIZE', sb_batch_size)
    script = script.replace('SB_FROM_IMAGENET', sb_from_imagenet)
    script = script.replace('SB_ROTATION', sb_rotation)
    script = script.replace('SB_EXPNAME', sb_expname)

    fname = f'{sb_jobname}{"_local" if gcp else ""}.sh'
    with open(f'training_scripts/{fname}', 'w') as f:
        f.write(script)

if __name__ == '__main__':

    GCP = False

    os.makedirs('training_scripts', exist_ok=True)
    # densenet121: 32
    # resnet50: 32
    # resnet18L 128

    BATCH_SIZE_MAP = {
        'resnet18': 24,
        'resnet50': 24, 
        'densenet121': 24, 
    }

    LR_EPOCH_MAP = {
        1e-5: 20,
        1e-4: 20,
        1e-2: 35
    }

    # for model in ['densenet121', 'resnet18', 'resnet50']:
    for model in ['resnet18']:
        for imagenet in [True, False]:
            for lr in [1e-5, 1e-4, 1e-2]:
                if not imagenet:
                    actual_lr = lr * 5
                else:
                    actual_lr = lr

                gen_script(model, actual_lr, BATCH_SIZE_MAP[model], imagenet, LR_EPOCH_MAP[lr], gcp=False)