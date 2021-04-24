"""Utility file for CUDA and GPU-specific functions."""


import torch
import torch.backends.cudnn as cudnn


def setup_gpus(gpu_ids):
    """Set up the GPUs and return the device to be used.

    Args:
        gpu_ids (list): list of GPU IDs

    Returns:
        device (str): the device, either 'cuda' or 'cpu'

    """
    device = None
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
        device = 'cuda'
    else:
        device = 'cpu'

    return device
