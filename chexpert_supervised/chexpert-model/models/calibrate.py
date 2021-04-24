import pickle
import numpy as np

from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration


class Calibrator(object):
    """Class for performing post-processing calibration techniques."""
    def __init__(self, calibrator_type, calibrator_dir, task_name, eval=True):
        # Where to save or load calibration model
        self.calibrator_type = calibrator_type
        self.path = calibrator_dir / (f"{calibrator_type}_{task_name}.pkl")
        self.eval = eval
        
        if self.eval:
            # If in eval mode, load the calibration model
            self.load()

    def predict(self, y_prob):
        # Run the loaded calibration model
        return self.calibrator.predict(y_prob)

    def train(self, y_true, y_prob):
        if self.calibrator_type == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.calibrator_type == 'platt':
            self.calibrator = _SigmoidCalibration()

        self.calibrator.fit(y_prob, y_true)

        self.save()

    def load(self):
        print(f"Loading calibration model from {self.path}")
        with self.path.open('rb') as f:
            self.calibrator = pickle.load(f)

    def save(self):
        print(f"Saving calibration model to {self.path}")
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)
        with self.path.open('wb') as f:
            pickle.dump(self.calibrator, f)
