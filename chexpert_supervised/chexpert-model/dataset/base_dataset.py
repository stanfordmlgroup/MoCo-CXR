import torchvision.transforms as transforms

from dataset.constants import CXR_MEAN, CXR_STD, IMAGENET_MEAN, IMAGENET_STD
from pathlib import Path
from torch.utils.data import Dataset
from .transforms import CLAHE
from .label_mapper import TASK_SEQUENCES, LabelMapper


class BaseDataset(Dataset):

    def __init__(self, data_dir, transform_args, split, is_training, dataset_name, tasks_to, dataset_task_sequence=None):
        """ Base class for CXR Dataset.
        Args:
            data_dir (string): Name of the root data director.
            transform_args (Namespace): Args for data transforms
            split (argsparse): Name of the dataset split to load (train, valid)
            dataset_name (string): Name of the dataset. Used to fetch the task sequence, used for this dataset.
                (the task sequence used when loading the csv)
            tasks_to (dict): Name of the sequence of tasks
                we want to map all our labels to.
        """

        assert isinstance(data_dir, str)
        assert isinstance(split, str)
        assert isinstance(dataset_name, str)
        assert isinstance(tasks_to, dict)

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.split = split
        self.is_training = is_training

        # Create a label mapper
        # Get the two label sequences as two dicts:
        # e.g {pathology1: 0, pathology2: 1...}
        if dataset_task_sequence is not None:
            self.original_tasks = TASK_SEQUENCES[dataset_task_sequence]
        else:
            self.original_tasks = TASK_SEQUENCES[dataset_name]
        self.target_tasks = tasks_to

        self.label_mapper = None

        if self.original_tasks != self.target_tasks:
            self.label_mapper = LabelMapper(
                    self.original_tasks,
                    self.target_tasks)

        self._set_transforms(transform_args)

    def _set_class_weights(self, labels):
        """Set class weights for weighted loss.

        Each task, gets its own set of class weights.

        Weights are calculate by taking 1 - the relative
        frequency of the class (positive vs negative)..

        Args:
            labels: Dataframe or numpy array containing
            a list of the labels. Shape should be
            (num_examples, num_labels)


        Example:
            100 examples with two tasks, cardiomegaly and consolidation.
            10 positve cases of cardiomegaly.
            20 positive cases of consolidation.

            We will then have:
            Class weights for cardiomegaly:
            [1-0.9, 1-0.1] = [0.1, 0.9]
            Class weights for consolidation:
            [1-0.8, 1-0.2] = [0.2, 0.8]

            The first element in each list is the wieght for the
            negative examples.
        """

        # Set weights for positive vs negative examples
        self.p_count = (labels == 1).sum(axis=0)
        self.n_count = (labels == 0).sum(axis=0)

        if self.label_mapper is not None:
            self.p_count = self.label_mapper.map(self.p_count)
            self.n_count = self.label_mapper.map(self.n_count)

        self.total = self.p_count + self.n_count

        self.class_weights = [self.n_count / self.total,
                        self.p_count / self.total]

    def _set_transforms(self, t_args):
        """Set the transforms

            Example:
                Image of size 1024x840.
                Scale to 312x256.
                Normalization and augmentation
                Random crop (or center crop) to 224x224.

            Note: Crop will be k * 224 and
            scale will be k*256.
        """

        # Shorter side scaled to t_args.scale
        if t_args.maintain_ratio:
            transforms_list = [transforms.Resize(t_args.scale)]
        else:
            transforms_list = [transforms.Resize((t_args.scale, t_args.scale))]

        # Data augmentation
        if self.is_training:
            transforms_list += [transforms.RandomHorizontalFlip() if t_args.horizontal_flip else None,
                                transforms.RandomRotation(t_args.rotate) if t_args.rotate else None,
                                transforms.RandomCrop((t_args.crop, t_args.crop)) if t_args.crop != 0 else None]
        else:
            transforms_list += [transforms.CenterCrop((t_args.crop, t_args.crop)) if t_args.crop else None]
        # Normalization
        if t_args.clahe:
            transforms_list += [CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))]

        if t_args.normalization == 'imagenet':
            normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        elif t_args.normalization == 'cxr_norm':
            normalize = transforms.Normalize(mean=CXR_MEAN, std=CXR_STD)
        transforms_list += [transforms.ToTensor(), normalize]

        self.transform = transforms.Compose([t for t in transforms_list if t])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        raise NotImplementedError
