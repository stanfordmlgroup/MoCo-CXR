"""Define Logger class for logging information to stdout and disk."""
import pandas as pd
import sys
from tensorboardX import SummaryWriter

from constants import COL_PATH, COL_TASK, COL_METRIC, COL_VALUE


class Logger(object):
    """Class for logging output."""
    def __init__(self, log_path, save_dir, results_dir=None):
        self.log_path = log_path
        self.log_file = log_path.open('w')

        self.tb_log_dir = save_dir / "tb"
        self.summary_writer = SummaryWriter(log_dir=str(self.tb_log_dir))

        self.results_dir = results_dir
        if results_dir is not None:
            self.metrics_path = results_dir / "scores.txt"
            self.metrics_csv_path = results_dir / "scores.csv"
            self.metrics_file = self.metrics_path.open('w')
            self.predictions_path = results_dir / "predictions.csv"
            self.groundtruth_path = results_dir / "groundtruth.csv"

    def log(self, *args):
        self.log_stdout(*args)
        print(*args, file=self.log_file)
        self.log_file.flush()

    def log_metrics(self, metrics, save_csv=False):
        for metric, value in metrics.items():
            msg = f'{metric}:\t{value}'
            if self.results_dir is not None:
                self.log_stdout(msg)
                print(msg, file=self.metrics_file)
                self.metrics_file.flush()
            else:
                self.log(f"[{msg}]")

        if save_csv:
            col_tasks = []
            col_metrics = []
            col_values = []
            for task_metric, value in metrics.items():
                # Extract task and metric from dict key
                tokens = task_metric.split(":")
                assert len(tokens) == 2, "Failed to split key on ':'!"
                task, metric = tokens
                col_tasks.append(task)
                col_metrics.append(metric)
                col_values.append(value)

            # Assemble a DataFrame and save as CSV
            metrics_df = pd.DataFrame({COL_TASK: col_tasks,
                                       COL_METRIC: col_metrics,
                                       COL_VALUE: col_values})
            metrics_df.to_csv(self.metrics_csv_path, index=False)

    def log_stdout(self, *args):
        print(*args, file=sys.stdout)
        sys.stdout.flush()

    def close(self):
        self.log_file.close()

    def log_scalars(self, scalar_dict, iterations, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.log_stdout(f'[{k}: {v:.3g}]')
            k = k.replace(':', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, iterations)

    # def log_scalars2(self, scalar_dict, iterations, print_to_stdout=True):
    #     """Log AUROC and accuracy in a dict as scalars to TensorBoard."""
    #     for k, v in scalar_dict.items():
    #         # Only prints AUROC and accuracy
    #         if ('AUROC' in k) or ('accuracy' in k):
    #             k = k.replace(':', '/')  # Group in TensorBoard by phase
    #             self.summary_writer.add_scalar(k, v, iterations)

    def log_predictions_groundtruth(self, predictions, groundtruth,
                                    paths=None):
        if paths is not None:
            predictions.insert(0, COL_PATH, paths)
            groundtruth.insert(0, COL_PATH, paths)

        predictions.to_csv(self.predictions_path, index=False)
        groundtruth.to_csv(self.groundtruth_path, index=False)

        if paths is not None:
            del predictions[COL_PATH]
            del groundtruth[COL_PATH]
