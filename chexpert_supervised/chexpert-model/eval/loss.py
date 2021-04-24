"""Define uncertainty cross entropy class."""
import torch
import torch.nn as nn

from constants import *


class CrossEntropyLossWithUncertainty(nn.Module):
    """Cross-entropy loss modified to also include uncertainty outputs."""
    def __init__(self, size_average=True, reduce=True):
        super(CrossEntropyLossWithUncertainty, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, logits, labels):
        """
        Args:
            logits: Un-normalized outputs of shape (batch_size, num_tasks, 3)
            labels: Labels of shape (batch_size, num_tasks)
                    where -1 is uncertain, 0 is negative, 1 is positive.
        """
        batch_size, last_dim = logits.size()
        if last_dim % 3:
            raise ValueError('Last dim should be divisible by 3, ' +
                             f'got last dim of {last_dim}')
        num_tasks = last_dim // 3

        # Fuse batch and task dimensions
        logits = logits.view(batch_size * num_tasks, 3)
        # Shift labels into range [0, 2]
        labels = (labels + 1).type(torch.int64)
        # Flatten
        labels = labels.view(-1)

        # Output shape (batch_size * num_tasks,)
        loss = self.ce_loss(logits, labels)
        # Reshape and take average over batch dim
        loss = loss.view(batch_size, num_tasks)

        if self.size_average:
            loss = loss.mean(1)
        if self.reduce:
            loss = loss.mean(0)

        return loss


class MaskedLossWrapper(nn.Module):

    def __init__(self, loss_fn, device):

        super().__init__()
        self.loss_fn = loss_fn
        self.device = device

    def _get_mask(self, targets):
        """Returns a mask to mask uncertain
        and missing labels.

        Functions tales advantage of the following:
            Negative/Positive: 0/1
            Uncertain: -1
            Missing: -2        """

        mask = torch.ones(targets.shape)
        mask[targets == UNCERTAIN] = 0
        mask[targets == MISSING] = 0

        mask = mask.to(self.device)

        return mask

    def forward(self, logits, targets):

        # Apply loss function
        loss = self.loss_fn(logits, targets)

        # Apply mask to skip missing labels
        # and handle uncertain labels
        mask = self._get_mask(targets)
        loss = loss * mask

        # Average the loss
        loss = loss.sum()
        loss = loss * (1 / (mask.sum()))

        return loss
