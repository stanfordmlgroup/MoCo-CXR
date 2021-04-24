import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

"""Map uncertain labels to constant labels (e.g. 0, 1) or to labels from another file."""
import argparse
import pandas as pd
import util

from collections import Counter, OrderedDict
from dataset.constants import COL_PATH, COL_SPLIT, COL_STUDY
from dataset import LabelMapper
from pathlib import Path
from tqdm import tqdm

STANFORD_TASKS = {
    "No Finding": 0,
    "Enlarged Cardiomediastinum": 1,
    "Cardiomegaly": 2,
    "Lung Lesion": 3,
    "Airspace Opacity": 4,
    "Edema": 5,
    "Consolidation": 6,
    "Pneumonia": 7,
    "Atelectasis": 8,
    "Pneumothorax": 9,
    "Pleural Effusion": 10,
    "Pleural Other": 11,
    "Fracture": 12,
    "Support Devices": 13
}


def main(args):
    dataset = MockDataset(args.data_dir, args.csv_name, args.map_name, args.constant_value, args.use_prevalences)

    print('Writing re-mapped labels to {}...'.format(args.output_path))
    dataset.labels.drop(columns="Study").to_csv(args.output_path, index=False)


def get_label2prevalence(df, tasks):

    label2prevalence = {}
    for task in tasks:
        num_labeled = ((df[task] == 1) | (df[task] == 0)).sum()
        num_positive = (df[task] == 1).sum()
        prevalence = num_positive / num_labeled

        label2prevalence[task] = prevalence

    return label2prevalence


class MockDataset(object):
    def __init__(self, data_dir, csv_name, map_name, constant_value=-1, use_prevalences=False):
        data_dir_path = Path(data_dir)
        csv_name_path = Path(csv_name)
        map_name_path = Path(map_name)

        self.original_tasks = OrderedDict(sorted(STANFORD_TASKS.items(), key=lambda x: x[1]))
        self.study_level = False
        self.uncertain_map_path = data_dir_path / map_name_path

        # Get labels and image paths
        self.labels = self._load_df(data_dir_path, csv_name_path, 'train', None, self.original_tasks)
        if use_prevalences:
            label2prevalence = get_label2prevalence(self.labels, self.original_tasks)
            self.labels = self._remap_with_dict(self.labels, label2prevalence)
        elif constant_value < 0:
            self.uncertain_map = self._load_df(None, map_name_path, 'train', None, self.original_tasks)
            self.labels = self._remap_with_file(self.labels, self.uncertain_map)
        else:
            self.labels = self._remap_with_constant(self.labels, constant_value)

        self.labels = self.labels.rename(columns={"Airspace Opacity": "Lung Opacity"})

    def _remap_with_file(self, labels_df, map_df):
        """Apply a mapping to uncertain labels (-1 in the dataset).

        Args:
            labels_df: Pandas DataFrame of labels (read-only, this function makes a copy).
            map_df: DataFrame with complete labels.
                Labels that are uncertain in the original labels will
                get overwritten with values from here.
        """

        # Copy and convert all labels to floats
        labels_df = labels_df.copy()
        labels_df[list(self.original_tasks)] = labels_df[list(self.original_tasks)].astype(float)

        # Overwrite all uncertain labels with values from the map spreadsheet
        num_mismatches = Counter()
        num_total = 0
        for col in labels_df:
            num_rows = len(labels_df[col])
            for i, row_value in tqdm(labels_df[col].iteritems(), total=num_rows):
                if row_value == LabelMapper.UNCERTAIN:# and labels_df[COL_SPLIT][i] != 'train-dev':
                    match_df = map_df.loc[map_df[COL_STUDY] == labels_df[COL_STUDY][i]]
                    num_total += 1
                    num_map_rows = len(match_df[col])
                    if num_map_rows != 1:
                        num_mismatches[num_map_rows] += 1
                        continue
                    labels_df.at[i, col] = match_df[col].values[0]

        print('Mismatches: {}'.format(num_mismatches))
        print('Had {} / {} mismatches during remapping'.format(sum(num_mismatches.values()), num_total))

        return labels_df

    def _remap_with_constant(self, labels_df, constant_value):
        """Apply a mapping to uncertain labels (-1 in the dataset).

        Args:
            labels_df: Pandas DataFrame of labels (read-only).
            constant_value: Value to overwrite uncertain labels.
        """
        # Copy and convert all labels to floats
        labels_df = labels_df.copy()
        labels_df[list(self.original_tasks)] = labels_df[list(self.original_tasks)].astype(float)

        # Overwrite all uncertain labels with values from the map spreadsheet
        for col in labels_df:
            num_rows = len(labels_df[col])
            for i, row_value in tqdm(labels_df[col].iteritems(), total=num_rows):
                if row_value == LabelMapper.UNCERTAIN:# and labels_df[COL_SPLIT][i] != 'train-dev':
                    labels_df.at[i, col] = constant_value

        return labels_df

    def _remap_with_dict(self, labels_df, map_dict):
        """Apply a mapping defined by a dictionary.

        Used to map labels to prevalences.

        Args:
            labels_df: Pandas DataFrame of labels (read-only).
            map_dict: Dictionary mapping column headers to constant values.
        """
        # Copy and convert all labels to floats
        labels_df = labels_df.copy()
        labels_df[list(self.original_tasks)] = labels_df[list(self.original_tasks)].astype(float)

        # Overwrite all uncertain labels with values from the map spreadsheet
        for col in labels_df:
            num_rows = len(labels_df[col])
            for i, row_value in tqdm(labels_df[col].iteritems(), total=num_rows):
                if row_value == LabelMapper.UNCERTAIN and col in map_dict:# and labels_df[COL_SPLIT][i] != 'train-dev':
                    labels_df.at[i, col] = map_dict[col]

        return labels_df

    @staticmethod
    def _load_df(data_dir, csv_name, split, subset, original_tasks):

        if data_dir is not None:
            csv_path = data_dir / csv_name
        else:
            csv_path = csv_name

        print('Loading DataFrame from {}...'.format(csv_path))
        df = pd.read_csv(csv_path)

        # Map Path -> Study (unique ID per study)
        df[COL_STUDY] = df.apply(lambda row: util.get_study_id(row[COL_PATH]), axis=1)

        df = df.rename(columns={"Lung Opacity": "Airspace Opacity"})

        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--constant_value', default=-1, type=int,
                        help='If non-negative, remap all uncertain labels to a constant value.')
    parser.add_argument('--use_prevalences', action='store_true',
                        help='If set, use prevalence of each pathology for uncertain labels.')
    # TODO: make this absolute path so csv_name and map_name does not
    # have to be inside data_dir
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--csv_name', default='master.csv', type=str,
                        help='Relative name. Assume file inside data_dir.')
    parser.add_argument('--map_name', default='output_probs_train.csv',
                        help='Relative name. Assume file inside data_dir.')
    parser.add_argument('--output_path', required=True, type=str,
                        help='Name of new csv file. Does not have to be' +
                        'inside data_dir.')

    main(parser.parse_args())
