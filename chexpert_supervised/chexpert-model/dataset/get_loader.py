import torch.utils.data as data

from .concat_dataset import ConcatDataset
from .su_dataset import SUDataset
from .nih_dataset import NIHDataset
from .label_mapper import TASK_SEQUENCES
from .pad_collate import PadCollate

def get_loader(data_args,
               transform_args,
               split,
               task_sequence,
               su_frac,
               nih_frac,
               batch_size,
               is_training=False,
               shuffle=False,
               study_level=False,
               frontal_lateral=False,
               return_info_dict=False):

    """Returns a dataset loader
       If both stanford_frac and nih_frac is one, the loader
       will sample both NIH and Stanford data.

    Args:
        stanford_frac: Float that specifies what percentage of stanford to load.
        nih_frac: Float that specifies what percentage of NIH to load.
        split: String determining if this is the train, valid, test, or sample split.
        shuffle: If true, the loader will shuffle the data.
        study_level: If true, creates a loader that loads the image on the study level.
            Only applicable for the SU dataset.
        frontal_lateral: If true, loads frontal/lateral labels.
            Only applicable for the SU dataset.
        return_info_dict: If true, return a dict of info with each image.

    Return:
        loader: A loader
    """

    if is_training:
        study_level=data_args.train_on_studies

    datasets = []
    if su_frac != 0:
        datasets.append(
                SUDataset(
                    data_args.su_data_dir,
                    transform_args, split=split,
                    is_training=is_training,
                    tasks_to=task_sequence,
                    frac=su_frac,
                    study_level=study_level,
                    frontal_lateral=frontal_lateral,
                    toy=data_args.toy,
                    return_info_dict=return_info_dict
                    )
                )

    if nih_frac != 0:
        datasets.append(
                NIHDataset(
                    data_args.nih_data_dir,
                    transform_args, split=split,
                    is_training=is_training,
                    tasks_to=task_sequence,
                    frac=nih_frac,
                    toy=data_args.toy
                    )
                )

    if len(datasets) == 2:
        assert study_level is False, "Currently, you can't create concatenated datasets when training on studies"
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    # Pick collate function
    if study_level:
        collate_fn = PadCollate(dim=0)
        loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8,
                             collate_fn=collate_fn)
    else:
        loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8)

    return loader


def get_eval_loaders(data_args, transform_args, task_sequence, batch_size, frontal_lateral, return_info_dict=False):
    """Returns a dataset loader
       If both stanford_frac and nih_frac is one, the loader
       will sample both NIH and Stanford data.

    Args:
        eval_su: Float that specifes what percentage of stanford to load.
        nih_frac: Float that specifes what percentage of NIH to load.
        args: Additional arguments needed to load the dataset.
        return_info_dict: If true, return a dict of info with each image.

    Return:
        loader: A loader

    """

    eval_loaders = []

    if data_args.eval_su:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'valid',
                                    task_sequence,
                                    su_frac=1,
                                    nih_frac=0,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    study_level=not frontal_lateral,
                                    frontal_lateral=frontal_lateral,
                                    return_info_dict=return_info_dict)]

    if data_args.eval_nih:
        eval_loaders += [get_loader(data_args,
                                    transform_args,
                                    'train',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=1,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    study_level=True,
                                    return_info_dict=return_info_dict),
                         get_loader(data_args,
                                    transform_args,
                                    'valid',
                                    task_sequence,
                                    su_frac=0,
                                    nih_frac=1,
                                    batch_size=batch_size,
                                    is_training=False,
                                    shuffle=False,
                                    study_level=True,
                                    return_info_dict=return_info_dict)]

    return eval_loaders

