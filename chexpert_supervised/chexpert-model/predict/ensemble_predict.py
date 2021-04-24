"""Define class for obtaining predictions from an ensemble."""
import json
import numpy as np
import pandas as pd

from .predict import Predictor
from saver import ModelSaver
from data import get_loader
from constants import *


class EnsemblePredictor(object):
    """Predictor class for an ensemble of models.

    Allows specification of different models per task.
    """

    def __init__(self, config_path, model_args, data_args,
                 gpu_ids, device, logger=None):
        """Instantiate an EnsemblePredictor object."""
        task2models, aggregation_fn = self.get_config(config_path)

        self.task2models = task2models
        self.aggregation_fn = aggregation_fn
        self.model_args = model_args
        self.data_args = data_args
        self.gpu_ids = gpu_ids
        self.device = device
        self.logger = logger

    def get_config(self, config_path):
        """Read configuration from a JSON file.

        Args:
            config_path: Path to configuration JSON file.

        Returns:
            task2models: Dictionary mapping task names to list of dicts.
                Each dict has keys 'ckpt_path' and 'model_uncertainty'.
            aggregation_fn: Aggregation function to combine predictions
                            from multiple models.
        """
        with open(config_path, 'r') as json_fh:
            self.config_dict = json.load(json_fh)
        task2models = self.config_dict[CFG_TASK2MODELS]
        agg_method = self.config_dict[CFG_AGG_METHOD]
        if agg_method == 'max':
            aggregation_fn = np.max
        elif agg_method == 'mean':
            aggregation_fn = np.mean
        else:
            raise ValueError(f'Invalid configuration: ' +
                             f'{CFG_AGG_METHOD} = {agg_method} ' +
                             '(expected "max" or "mean")')

        return task2models, aggregation_fn

    def save_config(self):
        """Save configuration file to run directory."""
        config_save_path = self.logger.results_dir / "config.json"
        self.logger.log(f"Saving config to {config_save_path}.")
        with open(config_save_path, 'w') as f:
            json.dump(self.config_dict, f, indent=4)

    def predict(self, cache=False, return_paths=False, all_gt_tasks=False):
        """Get model predictions on the evaluation set.

        Args:
            cache: Bool indicating whether to cache ensemble predictions.
                   If true, first tries to load already cached files,
                   then writes all predictions (and groundtruth) which
                   have not been cached.
            return_paths: Whether to also return corresponding study paths
            all_gt_tasks: Whether to return all ground truth columns


        Return:
            ensemble probabilities Pandas DataFrame,
            ensemble groundtruth Pandas DataFrame
        """
        is_cached = False
        if cache and self.logger is not None:
            results_dir = self.logger.results_dir
            self.predictions_path = results_dir / "ensemble_predictions.csv"
            self.groundtruth_path = results_dir / "groundtruth.csv"
            if (self.predictions_path.exists()
                    and self.groundtruth_path.exists()):
                self.logger.log(f"Predictions at {self.predictions_path} " +
                                "already exist. Loading from this file.")
                ensemble_probs_df = pd.read_csv(self.predictions_path)
                ensemble_gt_df = pd.read_csv(self.groundtruth_path)
                is_cached = True

        elif cache:
            raise ValueError("Must instantiate Predictor with logger" +
                             "if caching.")

        ensemble_paths = None
        if not is_cached:
            model2probs = {}
            model2gt = {}
            task2ensemble_probs = {}
            task2gt = {}
            self.save_config()
            for task, model_dicts in self.task2models.items():
                print('[======================]')
                print(task)
                for model_dict in model_dicts:
                    ckpt_path = Path(model_dict[CFG_CKPT_PATH])
                    is_3class = model_dict[CFG_IS_3CLASS]

                    if (ckpt_path in model2probs):
                        # We've already computed predictions for this model,
                        # skip it!
                        continue
                    ckpt_save_dir = Path(ckpt_path).parent
                    results_parent_dir = ckpt_save_dir / "results"
                    results_dir = results_parent_dir / self.data_args.phase
                    results_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_iter = ckpt_path.stem.split(".")[0]
                    predictions_name = f"{ckpt_iter}-predictions.csv"
                    groundtruth_name = f"{ckpt_iter}-groundtruth.csv"
                    predictions_path = results_dir / predictions_name
                    groundtruth_path = results_dir / groundtruth_name
                    if cache and (predictions_path.exists() and
                                  groundtruth_path.exists()):
                        self.logger.log(f"Predictions at {predictions_path}" +
                                        " already exist. Loading from this " +
                                        "file.")
                        probs_df = pd.read_csv(predictions_path,
                                               dtype=np.float32)
                        gt_df = pd.read_csv(groundtruth_path,
                                            dtype=np.float32)

                    else:
                        dataset = self.data_args.dataset
                        # Get model args from checkpoint and add them to
                        # command-line specified model args.
                        model_args, transform_args =\
                            ModelSaver.get_args(cl_model_args=self.model_args,
                                                dataset=dataset,
                                                ckpt_save_dir=ckpt_save_dir,
                                                model_uncertainty=is_3class)
                        model_args.moco = self.model_args.moco

                        model, ckpt_info =\
                            ModelSaver.load_model(ckpt_path=ckpt_path,
                                                  gpu_ids=self.gpu_ids,
                                                  model_args=model_args,
                                                  is_training=False)
                        predictor = Predictor(model=model, device=self.device)
                        loader = get_loader(phase=self.data_args.phase,
                                            data_args=self.data_args,
                                            transform_args=transform_args,
                                            is_training=False,
                                            return_info_dict=return_paths,
                                            logger=self.logger)

                        if loader.dataset.return_info_dict:
                            probs_df, gt_df, paths = predictor.predict(loader)
                            if ensemble_paths is None:
                                ensemble_paths = paths
                        else:
                            probs_df, gt_df = predictor.predict(loader)

                        if cache:
                            self.logger.log("Writing predictions to " +
                                            f"{predictions_path}.")
                            probs_df.to_csv(predictions_path, index=False)
                            self.logger.log("Writing groundtruth to " +
                                            f"{groundtruth_path}.")
                            gt_df.to_csv(groundtruth_path, index=False)

                    model2probs[ckpt_path] = probs_df
                    model2gt[ckpt_path] = gt_df

                task_ckpt_probs =\
                    [model2probs[Path(model_dict[CFG_CKPT_PATH])][task]
                     for model_dict in model_dicts]
                task2ensemble_probs[task] =\
                    self.aggregation_fn(task_ckpt_probs, axis=0)

                if len(model_dicts) > 0:
                    first_gt = model2gt[Path(model_dicts[0][CFG_CKPT_PATH])][task]
                    task2gt[task] =\
                        model2gt[Path(model_dicts[0][CFG_CKPT_PATH])][task]

            ensemble_probs_df = pd.DataFrame({task: task2ensemble_probs[task]
                                              for task in self.task2models if task in task2gt})
            if all_gt_tasks:
                ensemble_gt_df = model2gt[Path(model_dicts[0][CFG_CKPT_PATH])]
            else:
                ensemble_gt_df = pd.DataFrame({task: task2gt[task]
                                               for task in self.task2models})
            if cache:
                self.logger.log("Writing predictions to "
                                f"{self.predictions_path}.")
                ensemble_probs_df.to_csv(self.predictions_path,
                                         index=False)
                self.logger.log("Writing groundtruth to "
                                f"{self.groundtruth_path}.")
                ensemble_gt_df.to_csv(self.groundtruth_path,
                                      index=False)

        if return_paths:
            return ensemble_probs_df, ensemble_gt_df, ensemble_paths

        return ensemble_probs_df, ensemble_gt_df
