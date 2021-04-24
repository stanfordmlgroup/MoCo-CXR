"""Define optimizer class for guiding model training."""
import torch
import torch.optim as optim
from time import time

from eval import AverageMeter


def round_down(x, m):
    """Round x down to a multiple of m."""
    return int(m * round(float(x) / m))


class Optimizer(object):
    def __init__(self, parameters, optim_args, batch_size,
                 iters_per_print, iters_per_visual, iters_per_eval,
                 dataset_len, logger=None):

        self.optimizer = optim_args.optimizer
        self.lr = optim_args.lr
        self.lr_scheduler_name = optim_args.lr_scheduler
        self.sgd_momentum = optim_args.sgd_momentum
        self.weight_decay = optim_args.weight_decay
        self.sgd_dampening = optim_args.sgd_dampening
        self.lr_step = 0
        self.lr_decay_step = optim_args.lr_decay_step
        self.lr_patience = optim_args.lr_patience
        self.num_epochs = optim_args.num_epochs
        self.epoch = optim_args.start_epoch
        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.loss_meter = AverageMeter()

        # Current iteration in epoch
        # (i.e., # examples seen in the current epoch)
        self.iter = 0
        # Current iteration overall (i.e., total # of examples seen)
        self.global_step = round_down((self.epoch - 1) * dataset_len,
                                      batch_size)
        self.iter_start_time = None
        self.epoch_start_time = None

        self.iters_per_print = iters_per_print
        self.iters_per_visual = iters_per_visual
        self.iters_per_eval = iters_per_eval

        self.logger = logger

        self.set_optimizer(parameters)
        self.set_scheduler()

    def load_optimizer(self, ckpt_path, gpu_ids):
        """Load optimizer and LR scheduler state from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for loading the state dict.
        """
        device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        self.optimizer.load_state_dict(ckpt_dict['optimizer'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])

    def set_optimizer(self, parameters):
        """Set the PyTorch optimizer for params.

        Args:
            parameters: Iterator of network parameters to
                        optimize (i.e., model.parameters()).

        Returns:
            PyTorch optimizer specified by optim_args.
        """
        if self.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.lr,
                                       momentum=self.sgd_momentum,
                                       weight_decay=self.weight_decay,
                                       dampening=self.sgd_dampening)
        elif self.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters, self.lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer}')

    def set_scheduler(self):
        """Set the PyTorch scheduler which updates the learning rate
        for the optimizer."""
        if self.lr_scheduler_name is None:
            self.lr_scheduler = None
        elif self.lr_scheduler_name == 'step':
            self.lr_scheduler =\
                optim.lr_scheduler.StepLR(self.optimizer,
                                          step_size=self.lr_decay_step,
                                          gamma=self.lr_decay_gamma)
        elif self.lr_scheduler_name == 'multi_step':
            self.lr_scheduler =\
                optim.lr_scheduler.MultiStepLR(self.optimizer,
                                               milestones=self.lr_milestones,
                                               gamma=self.lr_decay_gamma)
        elif self.lr_scheduler_name == 'plateau':
            self.lr_scheduler =\
                optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                     factor=self.lr_decay_gamma,
                                                     patience=self.lr_patience,
                                                     min_lr=[pg['lr'] * 1e-3
                                                             for pg in self.optimizer.param_groups])
        else:
            raise ValueError('Invalid learning rate scheduler: ' +
                             f'{self.lr_scheduler_name}.')

    def step_scheduler(self, metric_value):
        """Step a LR scheduler.

        Args:
            metric_value: Metric value to determine the best checkpoint.
        """
        if self.lr_scheduler is not None:
            self.lr_step += 1

            if isinstance(self.lr_scheduler,
                          optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metric_value, epoch=self.lr_step)
            else:
                self.lr_scheduler.step(epoch=self.lr_step)

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch

    def start_epoch(self):
        self.epoch_start_time = time()
        self.iter = 0
        self.logger.log(f'[start of epoch {self.epoch}]')

    def end_epoch(self, metrics):
        self.epoch += 1

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def log_iter(self, inputs, logits, targets, unweighted_loss):
        """Log results from a training iteration."""
        loss = unweighted_loss.item()

        self.loss_meter.update(loss, inputs.size(0))

        # Periodically write to the log and TensorBoard
        if self.iter % self.iters_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = (f'[epoch: {self.epoch}, ' +
                       f'iter: {self.iter} / {self.dataset_len}, ' +
                       f'time: {avg_time:.2f}, ' +
                       f'loss: {self.loss_meter.avg:.3g}]')

            # Write all errors as scalars to the graph
            batch_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_scalars({'batch_lr': batch_lr},
                                    self.global_step,
                                    print_to_stdout=False)
            self.logger.log_scalars({'batch_loss': self.loss_meter.avg},
                                    self.global_step,
                                    print_to_stdout=False)
            self.loss_meter.reset()

            self.logger.log(message)

        # # Periodically visualize up to num_visuals
        # # training examples from the batch
        # if self.iter % self.iters_per_visual == 0:
        #     self.visualize(inputs, logits, targets, phase='train')
