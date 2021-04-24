import pandas as pd
import numpy as np
import sklearn.metrics as sk_metrics
import torch.nn as nn

from .below_curve_counter import BelowCurveCounter
from .loss import CrossEntropyLossWithUncertainty, MaskedLossWrapper


class Evaluator(object):
    """Evaluator class for evaluating predictions against
    binary groundtruth."""
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.kwargs = kwargs

        if "operating_points_path" in kwargs:
            self.rad_perf = pd.read_csv(kwargs["operating_points_path"])
        else:
            self.rad_perf = None

        self.set_eval_functions()

    def evaluate(self, groundtruth, predictions, metric, threshold=0.5):
        """Evaluate a single metric on groundtruth and predictions."""
        print("Evaluating metric: {}".format(metric))
        if metric in self.summary_metrics:
            metric_fn = self.summary_metrics[metric]
            value = metric_fn(groundtruth, predictions)
        elif metric in self.curve_metrics:
            metric_fn = self.curve_metrics[metric]
            value = metric_fn(groundtruth, predictions)
        elif metric in self.point_metrics:
            metric_fn = self.point_metrics[metric]
            value = metric_fn(groundtruth, predictions > threshold)
            # if metric == 'precision' or metric == 'recall':
            #    if value < 0.01:
            #        raise ValueError(f"Metric {metric} should not have score less than 0.01")
        else:
            raise ValueError(f"Metric {metric} not supported.")

        return value

    def evaluate_tasks(self, groundtruth, predictions, threshold=0.5):
        """Compute evaluation metrics and curves on multiple tasks."""
        metrics = {}
        curves = {}
        for task in list(predictions):
            print("Evaluating task: {}".format(task))

            task_groundtruth = groundtruth[task]
            task_predictions = predictions[task]
            # filter out those with -1 in groundtruth
            non_label = task_groundtruth.index[task_groundtruth == -1.0]
            task_predictions = task_predictions.drop(non_label)
            task_groundtruth = task_groundtruth.drop(non_label)

            metrics.update({f"{task}:{metric}":
                            self.evaluate(task_groundtruth,
                                          task_predictions,
                                          metric=metric)
                            for metric in self.summary_metrics})

            metrics.update({f"{task}:{metric}@thresh={threshold}":
                            self.evaluate(task_groundtruth,
                                          task_predictions,
                                          metric=metric,
                                          threshold=threshold)
                            for metric in self.point_metrics})
            """
            if self.rad_perf is not None:

                below_curve_counter = BelowCurveCounter(self.rad_perf,
                                                        task)
                metrics.update({
                    f'{task}:rads_below_ROC':
                    below_curve_counter.ROC(task_groundtruth,
                                            task_predictions),
                    f'{task}:rads_below_PR':
                    below_curve_counter.PR(task_groundtruth,
                                           task_predictions)
                })
            """
            curves.update({f"{task}:{metric}":
                           self.evaluate(task_groundtruth,
                                         task_predictions,
                                         metric=metric,
                                         threshold=threshold)
                           for metric in self.curve_metrics})

        return metrics, curves

    def evaluate_average_metric(self, metrics, evaluate_tasks,
                                average_metric_name):
        """Evaluate an average metric over classes."""

        # All provided names must be of the form "...-{metric_name}"
        metric_name = average_metric_name.split("-")[-1]

        average_metric = np.mean([metrics[f"{task}:{metric_name}"]
                                  for task in evaluate_tasks])

        return average_metric

    def set_eval_functions(self):
        """Set the evaluation functions."""
        def undefined_catcher(func, x, y):
            try:
                return func(x, y)
            except Exception:
                return np.nan

        # Functions that take probs as input
        self.summary_metrics = {
            'AUPRC': lambda x, y: undefined_catcher(sk_metrics.average_precision_score, x, y),
            'AUROC': lambda x, y: undefined_catcher(sk_metrics.roc_auc_score, x, y),
            'log_loss': lambda x, y: undefined_catcher(sk_metrics.log_loss, x, y),
        }

        # Functions that take binary values as input
        self.point_metrics = {
            'accuracy': lambda x, y: undefined_catcher(sk_metrics.accuracy_score, x, y),
            'precision': lambda x, y: undefined_catcher(sk_metrics.precision_score, x, y),
            'recall': lambda x, y: undefined_catcher(sk_metrics.recall_score, x, y),
        }

        self.curve_metrics = {
            'PRC': lambda x, y: undefined_catcher(sk_metrics.precision_recall_curve, x, y),
            'ROC': lambda x, y: undefined_catcher(sk_metrics.roc_curve, x, y),
        }

    def get_loss_fn(self, loss_fn_name, model_uncertainty,
                    mask_uncertain, device):
        """Get the loss function used for training.

        Args:
            loss_fn_name: Name of loss function to use.
            model_uncertainty: Bool indicating whether to predict
                               UNCERTAIN directly.
            mask_uncertain: Bool indicating whether to mask
                            UNCERTAIN labels.
            device: device to compute loss on (gpu or cpu).
        """
        print("evaluator: loss function name: {}".format(loss_fn_name))
        if model_uncertainty:
            loss_fn = CrossEntropyLossWithUncertainty()
        elif loss_fn_name == 'cross_entropy':
            loss_fn = nn.BCEWithLogitsLoss(reduction="none"
                                           if mask_uncertain else "mean")

            # Apply a wrapper that masks uncertain labels.
            if mask_uncertain:
                loss_fn = MaskedLossWrapper(loss_fn, device)

        else:
            raise ValueError("No loss function for supplied arguments.")

        return loss_fn
