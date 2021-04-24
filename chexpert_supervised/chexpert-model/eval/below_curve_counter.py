"""Define below curve counter class."""
import sklearn.metrics as sk_metrics


class BelowCurveCounter(object):
    def __init__(self, rad_perf, task_name):
        self.rad_perf = rad_perf
        self.task_name = task_name

    def ROC(self, ground_truth, predictions):

        self.rad_perf.index = self.rad_perf['Score']
        num_below_roc = 0

        fpr, tpr, threshold = sk_metrics.roc_curve(ground_truth, predictions)
        for rad_name in ['Rad1', 'Rad2', 'Rad3']:
            rad_sensitivity =\
                self.rad_perf.loc[f'{self.task_name} Sensitivity',
                                  rad_name]
            rad_specificity =\
                self.rad_perf.loc[f'{self.task_name} Specificity',
                                  rad_name]

            rad_vertical_projection, rad_horizontal_projection =\
                self._project(fpr, tpr, 1 - rad_specificity, rad_sensitivity)

            if (rad_vertical_projection >= rad_sensitivity):
                num_below_roc += 1

        return num_below_roc

    def PR(self, ground_truth, predictions):
        self.rad_perf.index = self.rad_perf['Score']

        num_below_pr = 0
        precision, recall, threshold =\
            sk_metrics.precision_recall_curve(ground_truth, predictions)

        for rad_name in ['Rad1', 'Rad2', 'Rad3']:
            rad_sensitivity =\
                self.rad_perf.loc[f'{self.task_name} Sensitivity',
                                  rad_name]
            rad_precision =\
                self.rad_perf.loc[f'{self.task_name} Precision',
                                  rad_name]

            rad_vertical_projection, rad_horizontal_projection =\
                self._project(recall, precision,
                              rad_sensitivity, rad_precision)

            if (rad_vertical_projection >= rad_precision):
                num_below_pr += 1

        return num_below_pr

    @staticmethod
    def _project(X, Y, rad_x, rad_y):
        """Find the closest points on the curve to the point in
        X and Y directions."""
        x = 0
        y = 0

        while (((x+2 < len(X)) and (X[x] > rad_x and X[x + 1] > rad_x))
                or (X[x] < rad_x and X[x + 1] < rad_x)):
            x += 1
        while ((y+2 < len(Y)) and (Y[y] > rad_y and Y[y + 1] > rad_y)
                or (Y[y] < rad_y and Y[y + 1] < rad_y)):
            y += 1

        rad_vertical_projection =\
            (Y[x + 1] - Y[x]) * (rad_x - X[x]) + Y[x]
        rad_horizontal_projection =\
            (X[y + 1] - X[y]) * (rad_y - Y[y]) + X[y]

        return rad_vertical_projection, rad_horizontal_projection
