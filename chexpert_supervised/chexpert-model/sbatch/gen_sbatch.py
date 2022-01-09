import os
from pathlib import Path
import math

DATA_CSV_ROOT = Path('/deep/group/data/moco/chexpert-proper-test-4/moving_logs')
SAVE_DIR = Path('~/CXR_RELATED/chexpert_save')

VALID_CSV = 'valid.csv'
TEST_CSV = 'test.csv'

# SEMI_RATIOS = [0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
# semi supervised ratio goes from 2e-9 to 2e0
SEMI_RATIOS = [0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
MODELS = ['resnet18', 'resnet50', 'densenet121']
MODELS_NAME_MAP = {'resnet18': 'ResNet18',
                   'resnet50': 'ResNet50',
                   'densenet121': 'DenseNet121'}

MODEL_SHORT_NAME_MAP = {'resnet18': 'r18',
                        'resnet50': 'r50',
                        'densenet121': 'd121'}

MOCO_IMAGENET_CKPTS = {'resnet18': 'YOUR RESNET18 (with image net) PATH',
                       'resnet50': 'YOUR RESNET50 (with image net)PATH',
                       'densenet121': 'YOUR DENSENET121 (with image net) PATH'}

MOCO_NO_IMAGENET_CKPTS = {'resnet18': 'YOUR RESNET18 (without image net) PATH',
                       'resnet50': 'YOUR RESNET50 (without image net) PATH',
                       'densenet121': 'YOUR DENSENET121 (without image net) PATH'}

START_LINES = \
"""#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="SB_JOBNAME"
#SBATCH --output=/sailhome/jingbo/CXR_RELATED/exp_logs/SB_JOBNAME-%j.out
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
cd ..;
"""


END_LINES = \
"""
# done
echo "Done"
"""


TRAIN_LINES = \
"""
python train.py \\
    --experiment_name SB_EXP_NAME \\
    --dataset chexpert_single  \\
    --model SB_MODEL \\
    --num_epochs SB_NUM_EPOCH \\
    --metric_name chexpert-competition-AUROC \\
    --train_custom_csv SB_TRAIN_CSV  \\
    --val_custom_csv SB_VAL_CSV \\
    --save_dir  SB_SAVE_DIR \\
    --pretrained SB_PRETRAINED \\
    --fine_tuning SB_FINE_TUNE \\
    --ckpt_path SB_PREV_MODEL_CKPT \\
    --iters_per_save SB_ITER_SAVE \\
    --iters_per_eval SB_ITER_EVAL
"""

CONFIG_LINES = \
"""
python select_ensemble.py \\
    --tasks "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion" \\
    --search_dir SB_CKPT_SEARCH
"""


TEST_LINES = \
"""
python test.py \\
    --dataset custom \\
    --moco false \\
    --phase test  \\
    --together true \\
    --ckpt_path SB_CKPT  \\
    --save_dir SB_TEST_SAVE \\
    --test_csv SB_TEST_CSV \\
    --test_image_paths SB_TEST_CSV \\
    --config_path SB_TEST_CONFIG
"""


def make_train_and_test(model, moco, imagenet, fine_tune, semi_ratio):
    sb_exp_name = f'{model}-{"moco" if moco else "baseline"}-{"wt" if imagenet else "wo"}-{fine_tune}-{semi_ratio}'

    # Handle training script
    sb_model = MODELS_NAME_MAP[model]
    sb_save_dir = str(SAVE_DIR)
    sb_train_csv = str(DATA_CSV_ROOT / (f'train_semi_{semi_ratio}.csv' if semi_ratio != 1 else 'train.csv'))
    sb_val_csv = str(DATA_CSV_ROOT / 'valid.csv')
    sb_prev_model_ckpt = None
    sb_pretrained = None
    sb_num_epoch = int(5 * 4 ** math.log(1 / semi_ratio, 8))
    sb_iter_save = min(int(2 ** 17 * semi_ratio), 8192)
    sb_iter_eval = sb_iter_save

    if fine_tune == 'full':
        sb_fine_tune = fine_tune
    else:
        if 'resnet' in model:
            sb_fine_tune = 'module.fc.weight,module.fc.bias'
        elif 'densenet' in model:
            sb_fine_tune = 'module.model.classifier.weight,module.model.classifier.bias'

    if moco:
        sb_pretrained = True
        if imagenet:
            sb_prev_model_ckpt = MOCO_IMAGENET_CKPTS[model]
        else:
            sb_prev_model_ckpt = MOCO_NO_IMAGENET_CKPTS[model]
    else:
        sb_prev_model_ckpt = "None"
        if imagenet:
            sb_pretrained = True
        else:
            sb_pretrained = False

    train_lines = TRAIN_LINES
    train_lines = train_lines.replace('SB_EXP_NAME', sb_exp_name)
    train_lines = train_lines.replace('SB_MODEL', sb_model)
    train_lines = train_lines.replace('SB_SAVE_DIR', sb_save_dir)
    train_lines = train_lines.replace('SB_TRAIN_CSV', sb_train_csv)
    train_lines = train_lines.replace('SB_VAL_CSV', sb_val_csv)
    train_lines = train_lines.replace('SB_NUM_EPOCH', str(sb_num_epoch))
    train_lines = train_lines.replace('SB_PREV_MODEL_CKPT', sb_prev_model_ckpt)
    train_lines = train_lines.replace('SB_PRETRAINED', str(sb_pretrained))
    train_lines = train_lines.replace('SB_FINE_TUNE', sb_fine_tune)
    train_lines = train_lines.replace('SB_ITER_SAVE', str(sb_iter_save))
    train_lines = train_lines.replace('SB_ITER_EVAL', str(sb_iter_eval))

    # Handle select checkpoint script
    sb_ckpt_search = str(SAVE_DIR / sb_exp_name)
    
    config_lines = CONFIG_LINES
    config_lines = config_lines.replace('SB_CKPT_SEARCH', sb_ckpt_search)

    # Handle testing script
    sb_test_csv = str(DATA_CSV_ROOT / 'test.csv')
    sb_test_config = str(SAVE_DIR / sb_exp_name / 'final.json')

    sb_ckpt = str(SAVE_DIR / sb_exp_name / 'best.pth.tar')
    sb_test_save = str(SAVE_DIR / sb_exp_name / 'test.pth.tar')

    test_lines = TEST_LINES
    test_lines = test_lines.replace('SB_CKPT', sb_ckpt)
    test_lines = test_lines.replace('SB_TEST_SAVE', sb_test_save)
    test_lines = test_lines.replace('SB_TEST_CSV', sb_test_csv)
    test_lines = test_lines.replace('SB_TEST_CONFIG', sb_test_config)

    return train_lines, config_lines, test_lines


def gen_sbatch(model, moco, imagenet, fine_tune):
    sb_file_name = f'sbatch-{model}-{"moco" if moco else "baseline"}-{"wt" if imagenet else "wo"}-{fine_tune}'

    sb_job_name = f'{MODEL_SHORT_NAME_MAP[model]}{"m" if moco else "b"}{"w" if imagenet else "o"}{fine_tune[0]}'
    fname = f'{sb_file_name}.sh'

    with open(fname, 'w') as f:

        start_line = START_LINES.replace('SB_JOBNAME', sb_job_name)
        f.write(start_line)
        f.write('\n')

        for semi_ratio in SEMI_RATIOS:

            f.write(f'# Semi Ratio = {semi_ratio}')
            f.write('\n')
            f.write('\n')

            train_lines, config_lines, test_lines = make_train_and_test(model, moco, imagenet, fine_tune, semi_ratio)

            f.write(train_lines)
            f.write('\n')
            
            f.write(config_lines)
            f.write('\n')

            f.write(test_lines)
            f.write('\n')

        f.write(END_LINES)
        f.write('\n')

    return fname


if __name__ == '__main__':

    COMBINATIONS = [(False, False, 'full'),     # scratch, baseline, full

                    (True, False, 'full'),      # imagenet, baseline, full
                    (True, True, 'full'),       # imagenet, moco, full
                    (False, True, 'full'),      # scratch, moco, full

                    (True, False, 'last'),      # imagenet, baseline, last
                    (True, True, 'last'),       # imagenet, moco, last
                    (False, True, 'last'),      # scratch, moco, last
                    ]

    
    for model in MODELS:

        commands = []

        for combo in COMBINATIONS:
            imagenet, moco, ft = combo

            fname = gen_sbatch(model, moco, imagenet, ft)
            commands.append(f'sbatch {fname}')


        with open(f'run_{model}.sh', 'w') as f:
            for c in commands:
                # if 'last' in c:
                #     c = f'# {c}'
                # print(c)
                f.write(c)
                f.write('\n')