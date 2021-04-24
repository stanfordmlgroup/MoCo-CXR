from .grad_cam import GradCAM

import torch
import numpy as np

class EnsembleCAM(object):
    """Class for generating CAMs using an ensemble."""
    def __init__(self, model, device):

        super(EnsembleCAM, self).__init__()

        self.device = device
        self.model = model

    def get_cam(self, x, task_id, task):

        ensemble_probs = []
        cams = []

        loaded_model_iterator = self.model.loaded_model_iterator(task)
        for loaded_model in loaded_model_iterator:
            grad_cam = GradCAM(loaded_model, self.device)
            probs = grad_cam.forward(x)

            grad_cam.backward(idx=task_id)

            cam = grad_cam.extract_cam()[0]

            ensemble_probs.append(probs)
            cams.append(cam)

        probs = self.model.aggregation_fn(ensemble_probs, axis=0)
        sorted_probs = np.sort(probs, axis=0)[::-1]
        idx = np.argsort(probs, axis=0)[::-1]

        cam = self.model.aggregation_fn(cams, axis=0)

        return sorted_probs, idx, cam