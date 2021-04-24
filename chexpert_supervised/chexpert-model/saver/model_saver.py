"""Define model saver class."""
import copy
import json
import queue
import shutil
import torch
import torch.nn as nn
from argparse import Namespace

import os
import models
from constants import *


class ModelSaver(object):
    """Class to save and load model ckpts."""
    def __init__(self, save_dir, iters_per_save, max_ckpts,
                 metric_name='val_loss', maximize_metric=False,
                 keep_topk=True, logger=None, **kwargs):
        """
        Args:
            save_dir: Directory to save checkpoints.
            iters_per_save: Number of iterations between each save.
            max_ckpts: Maximum number of checkpoints to keep before
                       overwriting old ones.
            metric_name: Name of metric used to determine best model.
            maximize_metric: If true, best checkpoint is that which
                             maximizes the metric value passed in via save.
            If false, best checkpoint minimizes the metric.
            keep_topk: Keep the top K checkpoints, rather than the most
                       recent K checkpoints.
        """
        super(ModelSaver, self).__init__()

        self.save_dir = save_dir
        self.iters_per_save = iters_per_save
        self.max_ckpts = max_ckpts
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_metric_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.keep_topk = keep_topk
        self.logger = logger

    def _is_best(self, metric_val):
        """Check whether metric_val is the best one we've seen so far."""
        if metric_val is None:
            return False
        return (self.best_metric_val is None
                or (self.maximize_metric and
                    self.best_metric_val < metric_val)
                or (not self.maximize_metric and
                    self.best_metric_val > metric_val))

    def save(self, iteration, epoch, model, optimizer, device, metric_val):
        """Save model parameters to disk.

        Args:
            iteration: Iteration that just finished.
            epoch: epoch to stamp on the checkpoint
            model: Model to save.
            optimizer: Optimizer for model parameters.
            device: Device where the model/optimizer parameters belong.
            metric_val: Value for determining whether checkpoint
                        is best so far.
        """
        lr_scheduler = None if optimizer.lr_scheduler is None\
            else optimizer.lr_scheduler.state_dict()
        ckpt_dict = {
            'ckpt_info': {'epoch': epoch, 'iteration': iteration,
                          self.metric_name: metric_val},
            'model_name': model.module.__class__.__name__,
            TASKS: model.module.tasks,
            'model_state': model.to('cpu').state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler
        }
        model.to(device)

        ckpt_path = self.save_dir / f'iter_{iteration}.pth.tar'
        print("Saver -- save checkpoint: {}".format(ckpt_path))
        if not os.path.exists(self.save_dir):
            print("Path {} not exists create new".format(self.save_dir))
            os.mkdir(self.save_dir)
        else:
            print("Path {} already exists".format(self.save_dir))
        torch.save(ckpt_dict, ckpt_path)

        if self._is_best(metric_val):
            # Save the best model
            if self.logger is not None:
                self.logger.log("Saving the model based on metric=" +
                                f"{self.metric_name} and maximize=" +
                                f"{self.maximize_metric} with value" +
                                f"={metric_val}.")
            self.best_metric_val = metric_val
            best_path = self.save_dir / 'best.pth.tar'
            shutil.copy(ckpt_path, best_path)

        # Add checkpoint path to priority queue (lower priority order gets
        # removed first)
        if not self.keep_topk:
            priority_order = iteration
        elif self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, ckpt_path))

        # Remove a checkpoint if more than max_ckpts ckpts saved
        if self.ckpt_paths.qsize() > self.max_ckpts:
            _, oldest_ckpt = self.ckpt_paths.get()
            try:
                oldest_ckpt.unlink()
            except OSError:
                pass

    @classmethod
    def load_ckpt_args(cls, ckpt_save_dir, dataset=None):
        """Load args from model ckpt.

        Args:
            ckpt_save_dir: pathlib directory pointing to model args.

        Returns:
            model_args: Namespace of model arguments read from ckpt_path.
            transform_args: Namespace of transform arguments
                            read from ckpt_path.
        """
        ckpt_args_path = ckpt_save_dir / 'args.json'
        with open(ckpt_args_path) as f:
            ckpt_args = json.load(f)

        model_args = ckpt_args['model_args']
        transform_args = ckpt_args['transform_args']

        if TASKS not in model_args and dataset is not None:
            model_args[TASKS] = DATASET2TASKS[dataset]

        return Namespace(**model_args), Namespace(**transform_args)

    @classmethod
    def get_args(cls, cl_model_args, dataset,
                 ckpt_save_dir, model_uncertainty):
        """Read args from ckpt_save_dir and make a new namespace combined with
        model_args from the command line."""
        model_args = copy.deepcopy(cl_model_args)
        ckpt_model_args, ckpt_transform_args =\
            cls.load_ckpt_args(ckpt_save_dir,
                               dataset)
        model_args.__dict__.update(ckpt_model_args.__dict__)
        model_args.__dict__.update({"model_uncertainty": model_uncertainty})

        return model_args, ckpt_transform_args

    @classmethod
    def load_model(cls, ckpt_path, gpu_ids, model_args, is_training=False):
        """Load model parameters from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for DataParallel.
            model_args: Model arguments to instantiate the model object.

        Returns:
            Model loaded from checkpoint, dict of additional
            checkpoint info (e.g. epoch, metric).
        """
        device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location=device)

        # import pdb; pdb.set_trace()

        # Build model, load parameters
        if not model_args.moco:
            model_fn = models.__dict__[ckpt_dict['model_name']]
        else:
            # TODO: this is how moco saves
            s = ckpt_dict['arch']

            # TODO: JBY
            if 'res' in s:
                s = s.replace('resnet', 'ResNet')
            elif 'dense' in s:
                s = s.replace('densenet', 'DenseNet')
            elif 'mnas' in s:
                s = s.replace('mnasnet', 'MNASNet')[:-3]
            model_fn = models.__dict__[s]

        if 'task_sequence' in ckpt_dict:
            tasks = list(ckpt_dict['task_sequence'].keys())
        elif TASKS in model_args.__dict__:
            tasks = model_args.__dict__[TASKS]
        else:
            if not model_args.moco:
                raise ValueError("Could not determine tasks.")
            else:
                tasks = CHEXPERT_TASKS

        print("Tasks: {}".format(tasks))
        model = model_fn(tasks, model_args)
        model = nn.DataParallel(model, gpu_ids)

        # TODO: JBY
        if not model_args.moco:
            model.load_state_dict(ckpt_dict['model_state'])
        else:
            state_dict = ckpt_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    # state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    state_dict['module.model.' + k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                elif 'encoder_k' in k or 'module.queue' in k:
                    del state_dict[k]
                elif k.startswith('module.encoder_q.fc'):
                    # if 'fc.0' not in k:
                    #     state_dict['module.model.fc' + k[len("module.encoder_q.fc.2"):]] = state_dict[k]
                    # TODO: JBY these are bad
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False)

        model = model.to(device)

        if is_training:
            model.train()
        else:
            model.eval()

        if model_args.moco:
            #moco
            ckpt_dict['ckpt_info'] = {}

        return model, ckpt_dict['ckpt_info']
