import argparse
import os
import random
import time
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

from .meters import AverageMeter
from .meters import ProgressMeter
from .combiner import detach_tensor

'''
def pred_accuracy(output, target, k):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    output = detach_tensor(output)
    target = detach_tensor(target)

    batch_size = target.size(0)

    argsorted_out = np.argsort(output)[:,-k:]
    return np.asarray(np.any(argsorted_y.T == target, axis=0).mean(dtype='f')),

    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]           # Seems like we only want the 1st
'''


def decorator_detach_tensor(function):
    def wrapper(*args, **kwargs):
        # TODO Find a simple way to handle this business ...
        # If is eval, or if fast debug, or
        # is train and not heavy, or is train and heavy
        output = detach_tensor(args[0])
        target = detach_tensor(args[1])
        args = args[2:]

        result = function(output, target, *args, **kwargs)
        return result
    return wrapper

@decorator_detach_tensor
def topk_acc(output, target, k):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    argsorted_out = np.argsort(output)[:,-k:]
    matching = np.asarray(np.any(argsorted_out.T == target, axis=0))
    return matching.mean(dtype='f')


@decorator_detach_tensor
def compute_auc_binary(output, target):
    #assuming output and target are all vectors for binary case
    try:
        o = softmax(output, axis=1)
        auc = roc_auc_score(target, o[:,1])
    except:
        return -1
    return auc


class Evaluator:

    def __init__(self, model, loss_func, metrics, loaders, args):

        self.model = model
        self.loss_func = loss_func
        self.metrics = metrics
        self.loaders = loaders
        self.args = args

        self.metric_best_vals = {metric: 0 for metric in self.metrics}


    def evaluate(self, eval_type, epoch):

        print(f'==> Evaluation for {eval_type}, epoch {epoch}')

        loader = self.loaders[eval_type]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        metric_meters = {metric: AverageMeter(metric, self.metrics[metric]['format']) \
                                                    for metric in self.metrics}
        list_meters = [metric_meters[m] for m in metric_meters]

        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, *list_meters],
            prefix=f'{eval_type}@Epoch {epoch}: ')

        # switch to evaluate mode
        self.model.eval()
        all_output = []
        all_gt = []

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                if self.args.gpu is not None:
                    images = images.cuda(self.args.gpu, non_blocking=True)
                target = target.cuda(self.args.gpu, non_blocking=True)
                all_gt.append(target.cpu())        

                # compute output
                output = self.model(images)
                all_output.append(output.cpu())
                
                loss = self.loss_func(output, target)
                
                # JBY: For simplicity do losses first
                losses.update(loss.item(), images.size(0))

                for metric in self.metrics:
                    args = [output, target, *self.metrics[metric]['args']]    
                    metric_func = globals()[self.metrics[metric]['func']]
                    result = metric_func(*args)
                    
                    metric_meters[metric].update(result, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #    .format(top1=top1, top5=top5))
            progress.display(i + 1)

        all_output = np.concatenate(all_output)
        all_gt = np.concatenate(all_gt)

        for metric in self.metrics:
            args = [all_output, all_gt, *self.metrics[metric]['args']]    
            metric_func = globals()[self.metrics[metric]['func']]
            result = metric_func(*args)
            
            metric_meters[metric].update(result, images.size(0))

            self.metric_best_vals[metric] = max(metric_meters[metric].avg,
                                                self.metric_best_vals[metric])

        progress.display(i + 1, summary=True)