"""Evaluate a ckpt or config on a test CSV.

Usage:
    python test_one.py --model_path <ckpt_path or config_path>
                       --csv_path <csv_path>
                       --name <unique name of experiment>

"""
import os
import pandas as pd
import sys

from argparse import ArgumentParser
from datetime import datetime
from getpass import getuser
from pathlib import Path
from shutil import copy
from subprocess import run


FILE_ENDINGS = set(['.pth', '.tar', '.json'])
ROOT_DIR = Path('/deep/group/chexperturbed/runs')
USER_DIR = ROOT_DIR / getuser()
TASKS = ['Cardiomegaly',
         'Edema',
         'Consolidation',
         'Atelectasis',
         'Pleural Effusion',
         'Normal']
METRIC = 'AUROC'


def parse_script_args():
    """Parse command line arguments.

    Returns:
        args (Namespace): parsed command line arguments

    """
    parser = ArgumentParser()

    parser.add_argument('--name', type=str, required=True,
                        help='Name of the run')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path of ckpt or config file')

    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to test_csv')

    parser.add_argument('--is_3class', action='store_true',
                        help='Whether this is a 3-class model')

    parser.add_argument('--save_cams', action='store_true',
                        help='Whether to also generate CAMs')

    parser.add_argument('--gpu_ids', type=str, required=True,
                        help='Devices to use')

    parser.add_argument('--inference_only', action='store_true',
                        help='Whether to only run inference')

    args = parser.parse_args()
    args.model_path = Path(args.model_path)
    assert args.model_path.exists()
    args.csv_path = Path(args.csv_path)
    assert args.csv_path.exists()
    if args.model_path.suffix not in FILE_ENDINGS:
        print('Error: unrecognized file ending! Exiting.')
        exit()
    return args


if __name__ == '__main__':
    args = parse_script_args()
    exp_dir = USER_DIR / args.name
    print('Saving run results in %s...' % str(exp_dir))
    USER_DIR.mkdir(exist_ok=True, parents=True)

    # Don't allow experiment to proceed if already exists, to avoid clobbering
    try:
        exp_dir.mkdir(parents=True)
    except FileExistsError as e:
        print('Error: directory already exists! Exiting.')
        exit()

    # Save command for reproducibility
    cmd_path = exp_dir / 'cmd.txt'
    print('Saving command to %s...' % str(cmd_path))
    cmd = ' '.join(['python'] + sys.argv)
    with open(cmd_path, 'w+') as f:
        f.write('%s\n' % cmd)

    # Testing ensemble
    model_path = None
    if args.model_path.suffix == '.json':
        config_dst_path = exp_dir / args.model_path.name
        copy(args.model_path, config_dst_path)
        model_path = ('--config_path', str(config_dst_path))
    # Testing single model
    else:
        ckpt_dst_path = exp_dir / args.model_path.name
        copy(args.model_path, ckpt_dst_path)
        model_path = ('--ckpt_path', str(ckpt_dst_path))
        args_dst_path = exp_dir / 'args.json'
        copy(args.model_path.parent / 'args.json', args_dst_path)

    test_args = ['python', 'test.py',
                 '--dataset', 'custom',
                 '--together', 'True',
                 '--test_csv', args.csv_path,
                 model_path[0], model_path[1],
                 '--phase', 'test',
                 '--save_dir', str(exp_dir),
                 '--gpu_ids', args.gpu_ids]

    if args.is_3class:
        test_args += ['--model_uncertainty', 'True']

    if args.save_cams:
        test_args += ['--save_cams', 'True']
        test_args += ['--only_competition_cams', 'True']

    if args.inference_only:
        test_args += ['--inference_only']

    # Run the model, but suppress output
    print('Running model...')
    with open(os.devnull, 'w') as devnull:
        run(test_args, stdout=devnull)

    # Delete the checkpoint when done to save space
    if model_path[0] == '--ckpt_path':
        print('Deleting checkpoint...')
        Path(model_path[1]).unlink()

    # Quit if we're only doing inference
    if args.inference_only:
        exit()

    # Print out relevant metrics
    scores_path = exp_dir
    if model_path[0] == '--config_path':
        scores_path /= args.model_path.stem
    scores_path = scores_path / 'results' / 'test' / 'scores.csv'
    df = pd.read_csv(scores_path)
    print('Selected results:')
    values = []
    for task in TASKS:
        value = float(df[(df['Metrics'] == METRIC) &
                         (df['Tasks'] == task)]['Values'])
        values.append(value)
        print('%s (%s): %f' % (METRIC, task, value))

    # Build row for spreadsheet
    ss_date = datetime.now().strftime('%m/%d/%Y')
    ss_path = str(args.model_path)
    ss_test_data = args.name.split('__')[-1]
    ss_values = [str(value) for value in values]
    ss_results_dir = str(exp_dir)
    ss_cmd = cmd
    ss_row = [ss_date, ss_path, ss_test_data]
    ss_row += ss_values + [ss_results_dir, ss_cmd]
    ss_row = ','.join(ss_row)
    print('Generated row for spreadsheet: %s' % ss_row)
    with open(exp_dir / 'row.txt', 'w+') as f:
        f.write('%s\n' % ss_row)
