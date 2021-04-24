import numpy as np
import json
import torch
import torch.nn.functional as F

from collections import OrderedDict
from .base_cam import BaseCAM


# Load the dictionary of model configs
# that for each model has the name of
# the last layer before the GAP
with open('cams/model_cam_configs.json') as f:
    MODEL_CONFIGS = json.load(f)


class GradCAM(BaseCAM):
    """Class for generating grad CAMs.
    Adapted from: https://github.com/kazuto1011/grad-cam-pytorch
    """
    def __init__(self, model, device):

        super(GradCAM, self).__init__(model, device)
        self.fmaps = OrderedDict()
        self.grads = OrderedDict()
        self.target_layer = MODEL_CONFIGS[model.module.__class__.__name__]['target_layer']

        def save_fmap(m, _, output):
            self.fmaps[id(m)] = output.to('cpu')

        def save_grad(m, _, grad_out):
            self.grads[id(m)] = grad_out[0].to('cpu')
	
        for name, module in self.model.named_modules():
            # Only put hooks on the target layer
            if name == self.target_layer:
                self.target_module_id = id(module)
                module.register_forward_hook(save_fmap)
                module.register_backward_hook(save_grad)

    def _find(self, outputs):

        # Since we've only put hooks on one layer
        # the target layer, we can return the value
        # right away
        return outputs[self.target_module_id]

    @staticmethod
    def _normalize(grads):
        return grads / (torch.norm(grads).item() + 1e-5)

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        weights = F.adaptive_avg_pool2d(grads, 1)
        return weights

    def extract_cam(self):
        """
            c: number of filters in final conv layer
            f: filter size
            shape of fmaps and grads : num_images x c x f x f
            shape of weights: num_images x c x 1 x 1
            shape of gcam: num_images x f x f
        """

        fmaps = self._find(self.fmaps)
        grads = self._find(self.grads)
        weights = self._compute_grad_weights(grads)

        assert len(fmaps.size()) == 4 and fmaps.size()[0] == 1


        assert len(weights.size()) == 4 and weights.size()[0] == 1

        # Sum up along the filter dimension
        gcam = (fmaps * weights).sum(dim=1)

        gcam = torch.clamp(gcam, min=0, max=float('inf'))

        gcam -= gcam.min()
        gcam /= (gcam.max() + 1e-7)

        return gcam.detach().to('cpu').numpy()


    def get_cam(self, x, task_id, task=None):
        
        probs = self.forward(x)
        sorted_probs = np.sort(probs, axis=0)[::-1]
        idx = np.argsort(probs, axis=0)[::-1]
        self.backward(idx=task_id)
        cam = self.extract_cam()[0]

        return sorted_probs, idx, cam



