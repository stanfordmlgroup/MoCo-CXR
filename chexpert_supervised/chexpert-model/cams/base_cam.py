import torch
import torch.nn.functional as F
import util


class BaseCAM(object):
    """Base class for generating CAMs.
    Adapted from: https://github.com/kazuto1011/grad-cam-pytorch
    """
    def __init__(self, model, device):
        super(BaseCAM, self).__init__()
        pred_type = '_3class' if model.module.model_uncertainty else 'binary'
        self.device = device
        self.pred_type = pred_type
        self.model = model
        self.model.eval()
        self.inputs = None

    def _encode_one_hot(self, idx):
        one_hot = torch.zeros([1, self.preds.size()[-1]],
                              dtype=torch.float32, device=self.device, requires_grad=True)

        if self.pred_type == '_3class':
            ind = 2 + idx * 3 # Get the index of positive class of the pathology.
            one_hot[0][ind] = 1.0
        else:
            one_hot[0][idx] = 1.0

        return one_hot

    def forward(self, x):
        self.inputs = x.to(self.device)
        self.model.zero_grad()
        self.preds = self.model(self.inputs)

        if self.pred_type == 'binary':
            self.probs = torch.sigmoid(self.preds)[0]
        elif self.pred_type == '_3class':
            self.probs = util.uncertain_logits_to_probs(self.preds)[0]
        else:
            self.probs = F.softmax(self.preds, dim=1)[0]
        return self.probs.detach().to('cpu').numpy()

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)

    def get_cam(self, x, task_id, task=None):
        raise NotImplementedError
