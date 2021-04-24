import cv2
import torch
import pandas as pd
from PIL import Image

import util
from .base_dataset import BaseDataset
from constants import *


class CheXpertDataset(BaseDataset):
    def __init__(self, csv_name, is_training, study_level,
                 transform_args, toy, return_info_dict, logger=None, data_args=None):
        # Pass in parent of data_dir because test set is in a different
        # directory due to dataset release, and uncertain maps are in a
        # different directory as well (both are under the parent directory).
        super().__init__(csv_name, is_training, transform_args)
        self.study_level = study_level
        self.toy = toy
        self.return_info_dict = return_info_dict
        self.logger = logger
        self.data_args = data_args

        self.is_train_dataset = self.csv_name == "train.csv"
        self.is_test_dataset = self.csv_name == "test.csv"
        self.is_val_dataset = self.csv_name == "valid.csv"
        self.is_uncertain_dataset = "uncertainty" in self.csv_name

        if self.is_test_dataset:
            self.csv_path = CHEXPERT_TEST_DIR / f"{csv_name}_image_paths.csv"
        elif self.is_uncertain_dataset:
            self.csv_path = CHEXPERT_UNCERTAIN_DIR / self.csv_name
        else:
            self.csv_path = CHEXPERT_DATA_DIR / self.csv_name
        
        if self.is_val_dataset:
            print("valid", self.csv_path)

        df = self.load_df()

        self.studies = df[COL_STUDY].drop_duplicates()

        if self.toy and self.csv_name == 'train.csv':
            self.studies = self.studies.sample(n=10)
            df = df[df[COL_STUDY].isin(self.studies)]
            df = df.reset_index(drop=True)

        # Set Study folder as index.
        if self.study_level:
            self.set_study_as_index(df)

        self.labels = self.get_labels(df)
        self.img_paths = self.get_paths(df)

    def load_df(self):
        df = pd.read_csv(Path(self.csv_path))

        # Prepend the data dir to get the full path.
        df[COL_PATH] = df[COL_PATH].apply(lambda x: CHEXPERT_PARENT_DATA_DIR / x)
        if self.is_test_dataset: #adjust for the fact that images are in codalab
            df[COL_PATH] = df[COL_PATH].apply(lambda p:
                                                Path(str(p).replace(str(CHEXPERT_DATA_DIR),
                                                                    str(CHEXPERT_TEST_DIR))))       
        df[COL_STUDY] = df[COL_PATH].apply(lambda p: Path(p).parent)
        if self.is_test_dataset:
            gt_df = pd.read_csv(CHEXPERT_TEST_DIR / "test_groundtruth.csv")
            gt_df[COL_STUDY] = gt_df[COL_STUDY].apply(lambda s: CHEXPERT_PARENT_DATA_DIR / s)
            gt_df[COL_STUDY] = gt_df[COL_STUDY].apply(lambda s: Path(str(s).replace(str(CHEXPERT_DATA_DIR),
                                                                               str(CHEXPERT_TEST_DIR))))
            df = pd.merge(df, gt_df, on=COL_STUDY, how = 'outer') 
            df = df.dropna(subset=['Path'])

        df = df.rename(columns={"Lung Opacity": "Airspace Opacity"}).sort_values(COL_STUDY)

        df[CHEXPERT_TASKS] = df[CHEXPERT_TASKS].fillna(value=0)

        return df

    def set_study_as_index(self, df):
        df.index = df[COL_STUDY]

    def get_paths(self, df):
        return df[COL_PATH]

    def get_labels(self, df):
        # Get the labels
        if self.study_level:
            study_df = df.drop_duplicates(subset=COL_STUDY)
            labels = study_df[CHEXPERT_TASKS]
        else:
            labels = df[CHEXPERT_TASKS]

        return labels

    def get_study(self, index):

        # Get study folder path
        study_path = self.studies.iloc[index]

        # Get and transform the label
        label = self.labels.loc[study_path].values
        label = torch.FloatTensor(label)

        # Get and transform the images
        # corresponding to the study at hand
        img_paths = pd.Series(self.img_paths.loc[study_path]).tolist()
        imgs = [Image.open(path).convert('RGB') for path in img_paths]
        # Downscale full resolution image to 1024 in the same way as 
        # performed in previous preprocessing, then convert back to PIL.
        # imgs = [util.resize_img(cv2.imread(str(path), 0), 1024) for path in img_paths]
        # imgs = [Image.fromarray(img).convert('RGB') for img in imgs]

        imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs)

        if self.return_info_dict:

            info_dict = {'paths': study_path}

            return imgs, label, info_dict

        return imgs, label

    def get_image(self, index):

        # Get and transform the label
        label = self.labels.iloc[index].values
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
            return self.get_study(index)
        else:
            return self.get_image(index)
