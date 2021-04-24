import numpy as np

import torchvision.transforms as t
from torch.utils.data import Dataset
from PIL import ImageEnhance

from constants import *


class BaseDataset(Dataset):
    def __init__(self, csv_name, is_training, transform_args):
        self.transform_args = transform_args
        self.csv_name = f"{csv_name}.csv" if not csv_name.endswith(".csv") else csv_name
        self.is_training = is_training

    def get_enhance_transform(self, f, enhance_min, enhance_max):
        def do_enhancement(img):
            factor = np.random.uniform(enhance_min, enhance_max)
            enhancer = f(img)
            return enhancer.enhance(factor)
        return do_enhancement
            
        
    def transform(self, img):
        """Set the transforms to be applied when loading."""

        transform_args = self.transform_args
        # Shorter side scaled to transform_args.scale
        if transform_args.maintain_ratio:
            transforms_list = [t.Resize(transform_args.scale)]
        else:
            transforms_list = [t.Resize((transform_args.scale, transform_args.scale))]

        # Data augmentation
        if self.is_training:
            if np.random.rand() < transform_args.rotate_prob:
                transforms_list += [t.RandomRotation((transform_args.rotate_min,
                                                      transform_args.rotate_max))]

            if np.random.rand() < transform_args.contrast_prob:
                transforms_list += [self.get_enhance_transform(ImageEnhance.Contrast,
                                                               transform_args.contrast_min,
                                                               transform_args.contrast_max)]

            if np.random.rand() < transform_args.brightness_prob:
                transforms_list += [self.get_enhance_transform(ImageEnhance.Brightness,
                                                               transform_args.brightness_min,
                                                               transform_args.brightness_max)]

            if np.random.rand() < transform_args.sharpness_prob:
                transforms_list += [self.get_enhance_transform(ImageEnhance.Sharpness,
                                                               transform_args.sharpness_min,
                                                               transform_args.sharpness_max)]

            if np.random.rand() < transform_args.horizontal_flip_prob:
                transforms_list += [t.Random.HorizontalFlip()]

            if transform_args.crop != 0:
                transforms_list += [t.RandomCrop((transform_args.crop, transform_args.crop))]

        else:
            transforms_list += [t.CenterCrop((transform_args.crop,
                                              transform_args.crop))
                                if transform_args.crop else None]

        if transform_args.normalization == 'imagenet':
            normalize = t.Normalize(mean=IMAGENET_MEAN,
                                    std=IMAGENET_STD)
        elif transform_args.normalization == 'chexpert_norm':
            normalize = t.Normalize(mean=CHEXPERT_MEAN,
                                    std=CHEXPERT_STD)
        transforms_list += [t.ToTensor(), normalize]

        return t.Compose([transform for transform in transforms_list if transform])(img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        raise NotImplementedError
