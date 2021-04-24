#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import training_tools.evaluator as eval_tools
from training_tools.meters import AverageMeter
from training_tools.meters import ProgressMeter

import aihc_utils.storage_util as storage_util
import aihc_utils.image_transform as image_transform

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
# JBY: Decrease number of workers
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


# Stanford AIHC modification
parser.add_argument('--exp-name', dest='exp_name', type=str, default='exp',
                    help='Experiment name')
parser.add_argument('--train_data', metavar='DIR',
                    help='path to train folder')
parser.add_argument('--val_data', metavar='DIR',
                    help='path to val folder')
parser.add_argument('--test_data', metavar='DIR',
                    help='path to test folder')
parser.add_argument('--save-epoch', dest='save_epoch', type=int, default=1,
                    help='Number of epochs per checkpoint save')
parser.add_argument('--from-imagenet', dest='from_imagenet', action='store_true',
                    help='use pre-trained ImageNet model')
parser.add_argument('--best-metric', dest='best_metric', type=str, default='acc@1',
                    help='metric to use for best model')
parser.add_argument('--semi-supervised', dest='semi_supervised', action='store_true',
                    help='allow the entire model to fine-tune')

parser.add_argument('--binary', dest='binary', action='store_true', help='change network to binary classif')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--cos-rate', default=4, type=float, metavar='CR',
                    help='Scaling factor for cos, higher the slower the decay')

parser.add_argument('--img-size', dest='img_size', type=int, default=320,
                    help='image size (Chexpert=320)')
parser.add_argument('--crop', dest='crop', type=int, default=320,
                    help='image crop (Chexpert=320)')
parser.add_argument('--maintain-ratio', dest='maintain_ratio', action='store_true',
                    help='whether to maintain aspect ratio or scale the image')
parser.add_argument('--rotate', dest='rotate', action='store_true',
                    help='to rotate image')
parser.add_argument('--optimizer', dest='optimizer', default='adam',
                    help='optimizer to use, chexpert=adam, moco=sgd')
parser.add_argument('--aug-setting', default='chexpert',
                    choices=['moco_v1', 'moco_v2', 'chexpert'],
                    help='version of data augmentation to use')
                    
best_metrics = {'acc@1': {'func': 'topk_acc', 'format': ':6.2f', 'args': [1]}}
                # 'acc@5': {'func': 'topk_acc', 'format': ':6.2f', 'args': [5]},
                #'auc': {'func': 'compute_auc_binary', 'format': ':6.2f', 'args': []}}
best_metric_val = 0


def main():

    args = parser.parse_args()
    print(args)
    checkpoint_folder = storage_util.get_storage_folder(args.exp_name, f'moco_lincls')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, checkpoint_folder))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, checkpoint_folder)


def main_worker(gpu, ngpus_per_node, args, checkpoint_folder):
    global best_metrics
    global best_metric_val
    if args.binary:
        best_metrics.update({'auc' : {'func': 'compute_auc_binary', 'format': ':6.2f', 'args': []}})
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=args.from_imagenet)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    num_classes = len(os.listdir(args.val_data)) #assume in imagenet format, so length == num folders/classes
    if num_classes == 2 and not args.binary:
        raise ValueError(f'Folder has {num_classes} classes, but you did not use "--binary" flag')
    elif num_classes != 2 and args.binary:
        raise ValueError(f'Folder has {num_classes} classes, but you used "--binary" flag')

    # init the fc layer
    if args.binary:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']

            # TODO JBY: Handle resume for current metrics setup
            raise NotImplementedError('Resuming not supported yet!')

            for metric in best_metrics:
                best_metrics[metric][0] = checkpoint[f'best_metrics'][metric]
            if args.gpu is not None:
                # best_acc_val may be from a checkpoint from a different GPU
                # best_acc_val = best_acc_val.to(args.gpu)
                # best_acc_test = best_acc_test.to(args.gpu)
                for metric in best_metrics:
                    best_metrics[metric][0] = best_metrics[metric][0].to(args.gpu)

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')

    traindir = args.train_data
    valdir = args.val_data
    testdir = args.test_data

    if args.aug_setting == 'moco_v2':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

        test_augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    elif args.aug_setting == 'chexpert':
        train_augmentation = image_transform.get_transform(args, training=True)
        test_augmentation = image_transform.get_transform(args, training=False)


    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(train_augmentation))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(test_augmentation)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose(test_augmentation)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    evaluator = eval_tools.Evaluator(model, criterion, best_metrics,\
                                     {'train': train_loader,\
                                      'valid': val_loader,\
                                      'test': test_loader}, args)

    if args.evaluate:
        evaluator.evaluate('valid', 0)
        evaluator.evaluate('test', 0)
        return
    
    evaluator.evaluate('test', 0)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, best_metrics)

        evaluator.evaluate('valid', epoch)
        evaluator.evaluate('test', 0)       # But we should technically not optimize for this

        is_best = evaluator.metric_best_vals[args.best_metric] > best_metric_val
        best_metric_val = max(best_metric_val, evaluator.metric_best_vals[args.best_metric])

        if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0 and \
             ((epoch % args.save_epoch == 0) or (epoch == args.epochs - 1))):
            save_checkpoint(checkpoint_folder, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_metrics': {metric: evaluator.metric_best_vals[metric] for metric in evaluator.metric_best_vals},
                'best_metric_val': best_metric_val,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            if epoch == args.start_epoch and args.pretrained:
                sanity_check(model.state_dict(), args.pretrained,
                             args.semi_supervised)

    evaluator.evaluate('test', epoch + 1)


def train(train_loader, model, criterion, optimizer, epoch, args, best_metrics):

    print(f'==> Training, epoch {epoch}')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    metric_meters = {metric: AverageMeter(metric,
                                          best_metrics[metric]['format']) \
                            for metric in best_metrics}
    list_meters = [metric_meters[m] for m in metric_meters]
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, *list_meters],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    # JBY: If semi-supervised, we tune on the entire model instead
    if args.semi_supervised:
        model.train()
    else:
        model.eval()
    all_output = []
    all_gt = []

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        all_gt.append(target.cpu().detach().numpy())

        # compute output
        output = model(images)
        all_output.append(output.cpu().detach().numpy())

        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        for metric in best_metrics:
            eval_args = [output, target, *best_metrics[metric]['args']]
            metric_func = eval_tools.__dict__[best_metrics[metric]['func']]
            result = metric_func(*eval_args)
            
            metric_meters[metric].update(result, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.display(i + 1)

    all_output = np.concatenate(all_output)
    all_gt = np.concatenate(all_gt)

    for metric in best_metrics:
        args = [all_output, all_gt, *best_metrics[metric]['args']]    
        metric_func = eval_tools.__dict__[best_metrics[metric]['func']]
        result = metric_func(*args)
        
        metric_meters[metric].update(result, images.size(0))
    progress.display(i + 1, summary=True)


def save_checkpoint(checkpoint_folder, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_folder, filename),
                        os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def sanity_check(state_dict, pretrained_weights, semi_supervised):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    if semi_supervised:
        print('SKIPPING SANITY CHECK for semi-supervised learning')
        return

    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


# JBY: Ported over support for Cosine learning rate
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        # TODO, JBY, is /4 an appropriate scale?
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs / args.cos_rate))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
