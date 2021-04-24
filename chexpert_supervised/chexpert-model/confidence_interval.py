import copy
import itertools
import numpy as np
import pandas as pd
import pathlib
import sklearn.metrics
import sys

import argparse

from constants import NamedTasks

class ConfidenceGenerator():
    # Confidence level is 0.95, then we do 1 - confidence level to get 0.05
    def __init__(self, confidence_level):
        self.records = []
        self.confidence_level = 1 - confidence_level 

    @staticmethod
    def compute_cis(series, confidence_level):
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(3)
        upper = sorted_perfs.iloc[upper_index].round(3)
        mean = sorted_perfs.mean().round(3)
        return lower, mean, upper

    def create_ci_record(self, perfs, name):
        lower, mean, upper = ConfidenceGenerator.compute_cis(
            perfs, self.confidence_level)
        record = {"name": name,
                  "lower": lower,
                  "mean": mean,
                  "upper": upper,
                  }
        self.records.append(record)

    def generate_cis(self, df):
        for diseases in df.columns:
            self.create_ci_record(df[diseases], diseases)

        df = pd.DataFrame.from_records(self.records)
        return df


def confidence(bootstraps, output_path, confidence_level=0.95):
    cb = ConfidenceGenerator(confidence_level=confidence_level)
    df = cb.generate_cis(bootstraps)

    df.to_csv(output_path, index=False)

def single_replicate_performances(gt, pred, diseases, metric, num_replicates):
    sample_ids = np.random.choice(len(gt), size=len(gt), replace=True)
    replicate_performances = {}
    gt_replicate = gt.iloc[sample_ids]
    pred_replicate = pred.iloc[sample_ids]

    for col in diseases:
        performance = metric(gt_replicate[col], pred_replicate[col])
        replicate_performances[col] = performance
    return replicate_performances

def multi_replicate_performances(gt, all_preds, diseases, metric, num_replicates):
    sample_ids = np.random.choice(len(gt), size=len(gt), replace=True)
    replicate_performances = {d: [None for i in range(len(all_preds))] for d in diseases}
    gt_replicate = gt.iloc[sample_ids]
    
    for i, pred in enumerate(all_preds):
        pred_replicate = pred.iloc[sample_ids]

        for col in diseases:
            performance = metric(gt_replicate[col], pred_replicate[col])
            replicate_performances[col][i] = performance

    averaged_rep_perf = {d: np.mean(replicate_performances[d]) for d in diseases}
    return averaged_rep_perf


def bootstrap_metric(gt, pred, all_preds, diseases, metric, num_replicates):
    
    all_performances = []
    all_multi_performances = []
    for _ in range(num_replicates):
        single_rep_performances = single_replicate_performances(gt, pred, diseases, metric, num_replicates)
        multi_rep_performances = multi_replicate_performances(gt, all_preds, diseases, metric, num_replicates)

        all_performances.append(copy.deepcopy(single_rep_performances))
        all_multi_performances.append(copy.deepcopy(multi_rep_performances))

    single_performances = pd.DataFrame.from_records(all_performances)
    multi_performances = pd.DataFrame.from_records(all_multi_performances)

    return single_performances, multi_performances


def compute_bootstrap_confidence_interval(gt, pred, all_preds,
                                          diseases, metric,
                                          num_replicates, confidence_level,
                                          output_path):
    single_bootstrap, multi_bootstrap = bootstrap_metric(gt, pred, all_preds,
                                                            diseases, metric,
                                                            num_replicates)

    confidence(single_bootstrap,
                output_path,
                confidence_level=0.95)
    confidence(multi_bootstrap,
                output_path.replace('.csv', '_multi.csv'),
                confidence_level=0.95)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Arguments for confidence_interval.py")
    parser.add_argument("--tasks", nargs='+', type=str)
    parser.add_argument("--custom_tasks", type=str)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--num_replicates", type=int, required=True)
    parser.add_argument("--confidence_level", type=float, required=True)
    parser.add_argument("--groundtruth", type=str, required=True)
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--num_splits", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # A redundant renaming of the arguments -- to avoid breaking the rest of the code.
    if args.custom_tasks is not None:
        disease_names = ','.join(NamedTasks[args.custom_tasks])
    else:
        disease_names = args.tasks
    metric_name = args.metric
    num_replicates = args.num_replicates
    confidence_level = args.confidence_level
    gt_path = args.groundtruth
    pred_path = args.prediction
    cur_iter = args.split
    num_iters = args.num_splits
    output_path = args.output

    print("Start confidence_interval...")
    # TODO JBY: Support more metrics
    assert metric_name == 'AUROC', 'Only AUROC is supported at the moment'

    diseases = disease_names.split(', ')
    diseases = [d.strip() for d in diseases]

    gt = pd.read_csv(gt_path)
    # gt = np.array(gt[disease_name].values.tolist())
    # gt = gt[disease_name]

    pred = pd.read_csv(pred_path)
    # pred = np.array(pred[disease_name].values.tolist())
    # pred = pred[disease_name]

    all_preds = []
    for i in range(num_iters):
        new_pred_path = pred_path.replace(f'it{cur_iter}', f'it{i}')
        all_preds.append(pd.read_csv(new_pred_path))

    # TODO, support more metrics

    print('Parsed arguments')

    compute_bootstrap_confidence_interval(
        gt, pred, all_preds, diseases, 
        sklearn.metrics.roc_auc_score,
        num_replicates, confidence_level,
        output_path)

    print('Confidence interval generated')
    
