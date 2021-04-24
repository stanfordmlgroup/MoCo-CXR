import torch.utils.data as data

from .chexpert_dataset import CheXpertDataset
from .custom_dataset import CustomDataset
from .pad_collate import PadCollate
from constants import *


def get_loader(phase, data_args, transform_args,
               is_training, return_info_dict,
               logger=None):
    """Get PyTorch data loader.

    Args:
        phase: string name of training phase {train, valid, test}.
        data_args: Namespace of data arguments.
        transform_args: Namespace of transform arguments.
        is_training: Bool indicating whether in training mode.
        return_info_dict: Bool indicating whether to return extra info
                          in batches.
        logger: Optional Logger object for printing data to stdout and file.

    Return:
        loader: PyTorch DataLoader object
    """

    study_level = not is_training
    shuffle = is_training

    # TODO: Make this more general
    if data_args.dataset == "chexpert":
        Dataset = CheXpertDataset
    elif 'special' in data_args.dataset:
        Dataset = CustomDataset
    elif data_args.dataset == "custom":
        Dataset = CustomDataset
    else:
        raise ValueError(f"Dataset {data_args.dataset} not supported.")

    # Get name of csv to load data from.
    # uncertain_map_path will replace this name.
    # need to make this more general!!!
    #csv_name = data_args.uncertain_map_path\
    #    if data_args.uncertain_map_path is not None else phase

    if data_args.uncertain_map_path is not None and phase == 'train':
        csv_name = data_args.uncertain_map_path
    else:
        csv_name = phase
    # Instantiate the Dataset class.
    dataset = Dataset(csv_name, is_training, study_level, transform_args,
                      data_args.toy, return_info_dict, logger, data_args)
    if study_level:
        # Pick collate function
        collate_fn = PadCollate(dim=0)
        loader = data.DataLoader(dataset,
                                 batch_size=data_args.batch_size,
                                 shuffle=shuffle,
                                 num_workers=data_args.num_workers,
                                 collate_fn=collate_fn)
    else:
        loader = data.DataLoader(dataset,
                                 batch_size=data_args.batch_size,
                                 shuffle=shuffle,
                                 num_workers=data_args.num_workers)

    return loader
