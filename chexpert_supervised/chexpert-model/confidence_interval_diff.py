from confidence_interval import *


def diff_replicate_performances(gt, all_preds1, all_preds2, diseases, metric, num_replicates):
    sample_ids = np.random.choice(len(gt), size=len(gt), replace=True)
    replicate_performances = {d: [None for i in range(len(all_preds1))] for d in diseases}
    gt_replicate = gt.iloc[sample_ids]
    
    #import pdb; pdb.set_trace()

    pred1_performances = {d: [None for i in range(len(all_preds1))] for d in diseases}
    for i, pred in enumerate(all_preds1):
        pred1_replicate = pred.iloc[sample_ids]
        for col in diseases:
            performance = metric(gt_replicate[col], pred1_replicate[col])
            pred1_performances[col][i] = performance
            #print(f'Pred 1[{i}] => {performance}')

    pred2_performances = {d: [None for i in range(len(all_preds1))] for d in diseases}
    for i, pred in enumerate(all_preds2):
        pred2_replicate = pred.iloc[sample_ids]
        for col in diseases:
            performance = metric(gt_replicate[col], pred2_replicate[col])
            pred2_performances[col][i] = performance
            #print(f'Pred 2[{i}] => {performance}')


    diff_rep_perf = {}
    for d in diseases:
        a1 = np.array(pred1_performances[d])
        a2 = np.array(pred2_performances[d])

        diff_rep_perf[d] = np.mean(a1 - a2)

    #import pdb; pdb.set_trace()
    # diff_rep_perf = {d: pred1_performances[d] - pred2_performances[d] for d in diseases}
    return diff_rep_perf


def bootstrap_diff_metric(gt, all_preds1, all_preds2, diseases, metric, num_replicates):
    
    all_multi_performances = []
    for _ in range(num_replicates):
        multi_rep_performances = diff_replicate_performances(
                gt, all_preds1, all_preds2, diseases, metric, num_replicates)

        all_multi_performances.append(copy.deepcopy(multi_rep_performances))

    multi_performances = pd.DataFrame.from_records(all_multi_performances)

    return multi_performances


def compute_bootstrap_diff_confidence_interval(gt, all_preds1, all_preds2,
                                          diseases, metric,
                                          num_replicates, confidence_level,
                                          output_path):
    multi_bootstrap = bootstrap_diff_metric(
                gt, all_preds1, all_preds2, diseases, metric, num_replicates)

    confidence(multi_bootstrap,
                output_path.replace('.csv', '_diff.csv'),
                confidence_level=0.95)


if __name__ == '__main__':
    # TODO: JBY: Big hack, no proper argparser used here!
    # Usage:
    #   python confidence_interval.py
    #       [DISEASE_NAME] [METRIC_NAME]
    #       [NUM_REPLICATES] [CONFIDENCE_LEVEL]
    #       [GT_CSV_PATH] 
    #       [PRED1_CSV_PATH]
    #       [PRED2_CSV_PATH]
    #       [CUR_ITER] [NUM_ITERS]
    #       [OUTPUT_PATH]

    assert len(sys.argv) == 11

    disease_names = sys.argv[1]
    metric_name = sys.argv[2]
    num_replicates = int(sys.argv[3])
    confidence_level = float(sys.argv[4])
    gt_path = sys.argv[5]
    pred1_path = sys.argv[6]
    pred2_path = sys.argv[7]
    cur_iter = int(sys.argv[8])
    num_iters = int(sys.argv[9])
    output_path = sys.argv[10]

    # TODO JBY: Support more metrics
    assert metric_name == 'AUROC', 'Only AUROC is supported at the moment'

    diseases = disease_names.split(', ')
    diseases = [d.strip() for d in diseases]

    gt = pd.read_csv(gt_path)
    # gt = np.array(gt[disease_name].values.tolist())
    # gt = gt[disease_name]

    pred1 = pd.read_csv(pred1_path)
    pred2 = pd.read_csv(pred2_path)
    # pred = np.array(pred[disease_name].values.tolist())
    # pred = pred[disease_name]

    all_preds1 = []
    for i in range(num_iters):
        new_pred_path = pred1_path.replace(f'it{cur_iter}', f'it{i}')
        all_preds1.append(pd.read_csv(new_pred_path))
    
    all_preds2 = []
    for i in range(num_iters):
        new_pred_path = pred2_path.replace(f'it{cur_iter}', f'it{i}')
        all_preds2.append(pd.read_csv(new_pred_path))

    # TODO, support more metrics

    print('Parsed arguments')

    compute_bootstrap_diff_confidence_interval(
        gt, all_preds1, all_preds2, diseases, 
        sklearn.metrics.roc_auc_score,
        num_replicates, confidence_level,
        output_path)

    print('Confidence interval generated')
    