from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
import os

from PIL import Image
from .base_dataset import BaseDataset
from .constants import COL_PATH, COL_STUDY


class SUDataset(BaseDataset):

    def __init__(self, data_dir,
                 transform_args, split, is_training,
                 tasks_to, study_level,
                 frontal_lateral=False, frac=1,
                 subset=None, toy=False,
                 return_info_dict=False):
        """ SU Dataset
        Args:
            data_dir (string): Name of the root data directory.
            transform_args (Namespace): Args for data transforms
            split (string): Name of the CSV to load.
            is_training (bool): True if training, False otherwise.
            tasks_to (string): Name of the sequence of tasks.
            study_level (bool): If true, each example is a study rather than an individual image.
            subset: String that specified as subset that should be loaded: AP, PA or Lateral.
            return_info_dict: If true, return a dict of info with each image.

        Notes:
            When study_level is true, the study folder is set as the index of the
            DataFrame. To retrieve images from a study, .loc[study_folder] is used.
            """

        dataset_task_sequence = 'stanford'

        super().__init__(data_dir, transform_args, split, is_training, 'stanford', tasks_to, dataset_task_sequence)

        self.subset = subset
        self.study_level = study_level
        self.return_info_dict = return_info_dict

        df = self._load_df(self.data_dir, split, subset, self.original_tasks)

        self.studies = df[COL_STUDY].drop_duplicates()

        if toy and split == 'train':
            self.studies = self.studies.sample(n=10)
            df = df[df[COL_STUDY].isin(self.studies)]
            df = df.reset_index(drop=True)

        # Sample a fraction of the data for training.
        if frac != 1 and is_training:
            self.studies = self.studies.sample(frac=frac)
            df = df[df[COL_STUDY].isin(self.studies)]
            df = df.reset_index(drop=True)

        # Set Study folder as index.
        if study_level:
            self._set_study_as_index(df)

        # Get labels and image paths.
        self.frontal_lateral = frontal_lateral
        self.labels = self._get_labels(df)
        self.img_paths = self._get_paths(df)

        # Set class weights.
        self._set_class_weights(self.labels)

    @staticmethod
    def _load_df(data_dir, split, subset, original_tasks):

        csv_name = f"{split}.csv" if not split.endswith(".csv") else split
        chexpert_data_dir = "CheXpert-v1.0"
        codalab_data_dir = "CodaLab"
        uncertainty_data_dir = "Uncertainty"

        if 'test' in split:
            
            csv_path = data_dir / codalab_data_dir / f"{split}_image_paths.csv"
            specific_data_dir = codalab_data_dir

        elif 'uncertainty' in split:

            csv_path = data_dir / uncertainty_data_dir / csv_name
            specific_data_dir = chexpert_data_dir

        else:

            csv_path = data_dir / chexpert_data_dir / csv_name
            specific_data_dir = chexpert_data_dir

        df = pd.read_csv(csv_path)
        df[COL_PATH] = df[COL_PATH].apply(lambda x: data_dir / x.replace(str(chexpert_data_dir), str(specific_data_dir)))
        df[COL_STUDY] = df[COL_PATH].apply(lambda p: str(p.parent))

        if 'test' in split:

            csv_name = "test_groundtruth.csv"
            gt_df = pd.read_csv(data_dir / codalab_data_dir / csv_name)

            gt_df[COL_STUDY] = gt_df[COL_STUDY].apply(lambda s: str(data_dir / s.replace(str(chexpert_data_dir), str(codalab_data_dir))))

            df = df.merge(gt_df, on=COL_STUDY) 

        df = df.rename(columns={"Lung Opacity": "Airspace Opacity"}).sort_values(COL_STUDY)

        df[list(original_tasks)] = df[list(original_tasks)].fillna(value=0)

        # Get PA, AP, or lateral.
        if subset is not None:

            if 'test' in split:
                raise ValueError('Test csv does not have metadata columns.')

            if subset in ['PA', 'AP']:
                df = df[df['AP/PA'] == subset]
            else:
                df = df[df['Frontal/Lateral'] == subset]

        return df


    @staticmethod
    def _set_study_as_index(df):
        df.index = df[COL_STUDY]

    @staticmethod
    def _get_paths(df):
        return df[COL_PATH]

    def _get_labels(self, df):

        # Get the labels
        if self.study_level:
            labels = df.drop_duplicates(subset=COL_STUDY)
            labels = labels[list(self.original_tasks)]
        elif self.frontal_lateral:
            labels = df[["Frontal/Lateral"]].apply(lambda x: x == "Lateral").astype(int)
        else:
            labels = df[list(self.original_tasks)]

        return labels

    def _get_study(self, index):

        # Get study folder path
        study_path = self.studies.iloc[index]

        # Get and transform the label
        label = np.array(self.labels.loc[study_path])
        if self.label_mapper is not None:
            label = self.label_mapper.map(label)
        label = torch.FloatTensor(label)

        # Get and transform the images
        # corresponding to the study at hand
        img_paths = pd.Series(self.img_paths.loc[study_path]).tolist()
        #imgs = [Image.open(path).convert('RGB') for path in img_paths]
        # Downscale full resolution image to 1024 in the same way as 
        # performed in previous preprocessing, then convert back to PIL.
        imgs = [resize_img(cv2.imread(str(path), 0), 1024) for path in img_paths]
        imgs = [Image.fromarray(img).convert('RGB') for img in imgs]

        imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs)

        if self.return_info_dict:

            info_dict = {'paths': study_path}

            return imgs, label, info_dict

        return imgs, label

    def _get_image(self, index):

        # Get and transform the label
        label = np.array(self.labels.iloc[index])
        if self.label_mapper is not None:
            label = self.label_mapper.map(label)
        label = torch.FloatTensor(label)

        # Get and transform the image
        img_path = self.img_paths.iloc[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.return_info_dict:
            info_dict = {'paths': str(img_path)}
            return img, label, info_dict

        return img, label

    def __getitem__(self, index):
        if self.study_level:
            return self._get_study(index)
        else:
            return self._get_image(index)

def resize_img(img, scale):
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    if max_ind == 0:
        # width fixed at scale
        wpercent = (scale / float(size[0]))
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # height fixed at scale
        hpercent = (scale / float(size[1]))
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)

    resized_img = cv2.resize(img, desireable_size[::-1])

    return resized_img


