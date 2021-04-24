"""Select models for an ensemble and assemble the corresponding JSON config.

Usage:
    Say [search_dir] is a directory containing multiple experiments, then:
    * To generate a config for an ensemble predicting Atelectasis and
      Pleural Effusion:
        python select_ensemble.py --search_dir [search_dir]
                                  --tasks "Atelectasis,Pleural Effusion"
    * To generate a config for all tasks, do not specify the --tasks arg:
        python select_ensemble.py --search_dir [search_dir]
    Configs are saved to [search_dir], under the default filename 'final.json'.

"""


import glob
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from sklearn.metrics import roc_auc_score

import util
from constants import CHEXPERT_TASKS
from data import get_loader
from predict import Predictor
from saver import ModelSaver

# TODO, JBY: add the following to handle model predictor hanging issue
import sys
import threading
from time import sleep
import _thread as thread

from constants import *

def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt    
    # raise TimeoutError


def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer    


def find_checkpoints(search_dir, ckpt_pattern):
    """Recursively search search_dir, and find all ckpts matching the pattern.

    When searching, the script will skip over checkpoints for which a
    corresponding args.json does not exist. It will also ensure that all
    models were validated on the same validation set.

    Args:
        search_dir (Path): the directory in which to search
        ckpt_pattern (str): the filename pattern to match

    Returns:
        ckpts (list): list of (Path, dict) corresponding to checkpoint paths
            and the corresponding args
        csv_dev_path (str): path specifying the validation set

    """
    # Temporarily switch to search_dir to make searching easier
    cwd = os.getcwd()
    os.chdir(search_dir)
    ckpts = []
    csv_dev_path = None

    # Recursively search for all files matching pattern
    for filename in glob.iglob('**/%s' % ckpt_pattern, recursive=True):
        ckpt_path = search_dir / filename
        ckpt_dir = ckpt_path.parent
        args_path = ckpt_dir / 'args.json'
        if not args_path.exists():
            print('args.json not found for %s! Skipping.' % str(ckpt_path))
            continue

        with open(args_path) as f:
            ckpt_args = json.load(f)

        assert 'csv_dev' in ckpt_args['data_args']
        ckpt_csv_dev_path = Path(ckpt_args['data_args']['csv_dev'])

        # Store csv_dev_path and make sure all validation sets are the same
        if csv_dev_path is None:
            csv_dev_path = ckpt_csv_dev_path
        else:
            assert csv_dev_path == ckpt_csv_dev_path
        ckpts.append((ckpt_path, ckpt_args))

    # Switch back to original working directory
    os.chdir(cwd)
    print('Found %d checkpoint(s).' % len(ckpts))
    return ckpts, csv_dev_path

@exit_after(1800)
def run_model(ckpt_path, ckpt_args, has_gpu, custom_tasks=None):
    """Run a model with the specified args and output predictions.

    Args:
        ckpt_path (Path): path specifying the checkpoint
        ckpt_args (dict): args associated with the corresponding run

    Returns:
        pred_df (pandas.DataFrame): model predictions
        gt_df (pandas.DataFrame): corresponding ground-truth labels

    """
    ckpt_save_dir = ckpt_path.parent
    model_args = Namespace(**ckpt_args['model_args'])
    # JBY: The samed model will not be moco
    model_args.moco = False
    transform_args = Namespace(**ckpt_args['transform_args'])
    data_args = Namespace(**ckpt_args['data_args'])
    print("in select_ensemble.py: data_args: {}".format(data_args))
    data_args.custom_tasks = custom_tasks

    if has_gpu:
        gpu_ids = util.args_to_list(ckpt_args['gpu_ids'], allow_empty=True,
                                    arg_type=int, allow_negative=False)
    else:
        # TODO: JBY: HACK! CHANGING GPU ID TO NONE
        gpu_ids = []
    device = util.setup_gpus(gpu_ids)
    model, _ = ModelSaver.load_model(ckpt_path=ckpt_path,
                                     gpu_ids=gpu_ids,
                                     model_args=model_args,
                                     is_training=False)
    predictor = Predictor(model=model, device=device)
    loader = get_loader(phase='valid',
                        data_args=data_args,
                        transform_args=transform_args,
                        is_training=False,
                        return_info_dict=False,
                        logger=None)
    pred_df, gt_df = predictor.predict(loader)
    return pred_df, gt_df


def get_auc_metric(task):
    """Get a metric that calculates AUC for a specified task.

    Args:
        task (str): the column over which to calculate AUC

    Returns:
        metric (function): metric operating on (pred_df, gt_df) to calculate
            AUC for the specified task

    """
    def metric(pred_df, gt_df):
        # AUC score requires at least 1 of each class label
        if len(set(gt_df[task])) < 2:
            return None
        return roc_auc_score(gt_df[task], pred_df[task])
    return metric


def rank_models(ckpt_path2dfs, metric, maximize_metric):
    """Rank models according to the specified metric.

    Args:
        ckpt_path2dfs (dict): mapping from ckpt_path (str) to (pred_df, gt_df)
        metric (function): metric to be optimized, computed over two
            DataFrames, namely the predictions and ground truth labels
        maximize_metric (bool): whether higher values of the metric are better
            (as opposed to lower values)

    Returns:
        ranking (list): list containing (Path, float), corresponding to
            checkpoint-metric pairs ranked from best to worst by metric value

    """
    assert len(ckpt_path2dfs)
    ranking = []
    values = []
    for ckpt_path, (pred_df, gt_df) in ckpt_path2dfs.items():
        try:
            value = metric(pred_df, gt_df)
            print(f'Computed {value}')
            if value is None:
                continue
            ranking.append((ckpt_path, value))
            values.append(value)
        except ValueError:
            continue

    # import pdb; pdb.set_trace()

    # For deterministic results, break ties using checkpoint name.
    # We can do this since sort is stable.
    ranking.sort(key=lambda x: x[0])
    ranking.sort(key=lambda x: x[1], reverse=maximize_metric)
    return ranking


def get_config_list(ranking, ckpt_path2is_3class):
    """Assemble a model list for a specific task based on the ranking.

    In addition to bundling information about the ckpt_path and whether to
    model_uncertainty, the config_list also lists the value of the metric to
    aid debugging.

    Args:
        ranking (list): list containing (Path, float), corresponding to
            checkpoint-metric pairs ranked from best to worst by metric value
        ckpt_path2is_3class (dict): mapping from ckpt_path to is_3class
            (whether to model_uncertainty)

    Returns:
        config_list (list): list bundling information about ckpt_path,
            model_uncertainty, and metric value

    """
    config_list = []
    for ckpt_path, value in ranking:
        is3_class = ckpt_path2is_3class[ckpt_path]
        ckpt_info = {'ckpt_path': str(ckpt_path),
                     'is_3class': is3_class,
                     'value': value}
        config_list.append(ckpt_info)
    return config_list


def assemble_config(aggregation_method, task2models):
    """Assemble the entire config for dumping to JSON.

    Args:
        aggregation_method (str): the aggregation method to use during ensemble
            prediction
        task2models (dict): mapping from task to the associated config_list of
            models

    Returns:
        (dict): dictionary representation of the ensemble config, ready for
            dumping to JSON

    """
    return {'aggregation_method': aggregation_method,
            'task2models': task2models}


def parse_script_args():
    """Parse command line arguments.

    Returns:
        args (Namespace): parsed command line arguments

    """
    parser = ArgumentParser()

    parser.add_argument('--search_dir',
                        type=str,
                        required=True,
                        help='Directory in which to search for checkpoints')

    parser.add_argument('--ckpt_pattern',
                        type=str,
                        default='iter_*.pth.tar',
                        help='Pattern for matching checkpoint files')

    parser.add_argument('--max_ckpts',
                        type=int,
                        default=30,
                        help='Max. number of checkpoints to select per task')

    parser.add_argument('--tasks',
                        type=str,
                        help='Prediction tasks of interest')

    parser.add_argument('--aggregation_method',
                        type=str,
                        default='mean',
                        help='Aggregation method to specify in config')

    parser.add_argument('--config_name',
                        type=str,
                        default='final.json',
                        help='Name for output JSON config')

    parser.add_argument('--has_gpu',
                        type=str,
                        default=True,
                        help='Whether evaluation has GPU')

    parser.add_argument('--custom_tasks',
                        type=str,
                        help='Use customized tasks to stay consistent with other train and test.')
    args = parser.parse_args()

    # If no task is specified, build config for all CheXpert tasks
    if args.custom_tasks is not None:
        args.tasks = NamedTasks[args.custom_tasks]
    elif args.tasks is None:
        args.tasks = CHEXPERT_COMPETITION_TASKS
    else:
        args.tasks = util.args_to_list(args.tasks, allow_empty=True,
                                       arg_type=str)
    return args


if __name__ == '__main__':
    args = parse_script_args()
    search_dir = Path(args.search_dir)

    print("Start select_ensemble...")
    # Retrieve all checkpoints that match the given pattern
    ckpts, csv_dev_path = find_checkpoints(search_dir, args.ckpt_pattern)

    # Get predictions for all checkpoints that were found
    ckpt_path2dfs = {}
    ckpt_path2is_3class = {}
    task2models = {}
    for i, (ckpt_path, ckpt_args) in enumerate(ckpts):
        print('Evaluating checkpoint (%d/%d).' % (i + 1, len(ckpts)))

        # try:
        pred_df, gt_df = run_model(ckpt_path, ckpt_args, args.has_gpu, args.custom_tasks)
        ckpt_path2dfs[ckpt_path] = (pred_df, gt_df)
        is_3class = ckpt_args['model_args']['model_uncertainty']
        ckpt_path2is_3class[ckpt_path] = is_3class
    # except:
        #    print(f'Evaluation for {i + 1} timed out')

    # Calculate task-specific metrics and rank the checkpoints
    for task in args.tasks:
        print('Ranking checkpoints for task "%s".' % task)
        metric = get_auc_metric(task)
        ranking = rank_models(ckpt_path2dfs, metric, maximize_metric=True)
        ranking = ranking[:min(args.max_ckpts, len(ranking))]
        task2models[task] = get_config_list(ranking, ckpt_path2is_3class)

    # Assemble and write the ensemble config file
    print('Writing config file to %s.' % str(search_dir / args.config_name))
    config = assemble_config(args.aggregation_method, task2models)
    with open(search_dir / args.config_name, 'w+') as f:
        json.dump(config, f, indent=4)
