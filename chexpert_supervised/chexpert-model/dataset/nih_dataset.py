from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .base_dataset import BaseDataset

class NIHDataset(BaseDataset):

    def __init__(self, data_dir,
                 transform_args, split, is_training, tasks_to, frac, toy=False):
        """ NIH Dataset
        Args:
            data_dir (string): Name of the root data director.
            transform_args (Namespace): Namespace object containing all the transform arguments.
            split (argsparse): Arguments used for transforms
            tasks_to (dict): The sequence of tasks.
        """

        super().__init__(data_dir, transform_args,
                split, is_training, 'nih', tasks_to)

        self.study_level = False

        # Load data from csv
        df = self._load_df(self.data_dir, split)
        if toy and split == 'train':
            df = df.sample(n=20)
            df = df.reset_index(drop=True)

        if frac != 1 and is_training:
            df = df.sample(frac=frac)
            df = df.reset_index(drop=True)

        # Get labels and studies
        self.labels = self._get_labels(df)

        # Get image paths
        self.img_paths = self._get_paths(df)

        # Set transforms and class weights
        self._set_class_weights(self.labels)

    @staticmethod
    def _load_df(data_dir, split):

        if split == 'test':
            csv_path = data_dir / 'test420.csv'
        else:
            csv_path = data_dir / (split + '_medium.csv')

        df = pd.read_csv(csv_path)
        img_dir = data_dir / 'images'
        df['Path'] = df['Path'].apply(lambda x: img_dir / x)
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def _get_paths(df):
        """Get list pf paths to images"""

        # Skip the first header row
        return df['Path'].tolist()[1:]

    def _get_studies(self, df):
        """The NIH dataset does not have study level data"""
        return None
    def _get_labels(self, df):
        """Return all the labels.

        In the NIH datset all labels are in one column. The
        diferent pathologies are separated with pipes
        E.g: 0|0|1|1|0|1|1|0|0|0|0|1|0|0
        """

        labels = np.array([np.fromstring(row['Label'], sep='|', dtype=int) for i, row in df.iterrows() if i])
        return labels

    def __getitem__(self, index):

        # Get and transform the label
        label = self.labels[index, :]
        if self.label_mapper is not None:
            label = self.label_mapper.map(label)
        label = torch.FloatTensor(label)

        # Get and transform the image
        img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.transform(img)

        return img, label
