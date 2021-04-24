import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) == torch.Tensor:
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def str_val(self):
        if self.name == 'Loss':
            fmtstr = '{name} {val' + self.fmt + '}\n'
        else:
            fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def __str__(self):
        if self.name == 'Loss':
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})\n'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, summary=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        if not summary:
            entries += [str(meter) for meter in self.meters]
            print('\t'.join(entries))
        else:
            entries += [meter.str_val() for meter in self.meters]
            print('Summary: ' + '\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

