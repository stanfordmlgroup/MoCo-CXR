'''File created to reorganize montgomery and shenzhen dataset to fit 
torchvision.ImageFolder class
'''

from collections import defaultdict
import copy
import os
import pprint as pp
import random
import re
import shutil
import sys

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 2e-7 ~ 2--1
ALL_SEMI_RATIO =  [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]

SEMI_ITERATIONS = { 0.0078125: 12,
                    0.015625: 10,
                    0.03125: 8,
                    0.0625: 8,
                    0.125: 4,
                    0.25: 4,
                    0.5: 4,
                    1: 1
                }

TEST_RATIO = 0.20
VAL_RATIO = 0.15
TOTAL_TRAIN_RATIO = 1 - TEST_RATIO - VAL_RATIO
TOTAL = 662


def verify_one_split(df, name, ratio=None):
    if df is None:
        return False

    total_len = len(df)

    if ratio is not None:
        if not (TOTAL * ratio > total_len * 0.95 and TOTAL * ratio < total_len * 1.05):
            print(f'Split {name} has incorrect number of items {total_len}')
            return False

    if 'Tuberculosis' not in df:
        return False

    # no_finding = len(df[df['No Finding'] == 0])
    tb = len(df[df['Tuberculosis'] == 1])
    no_tb = len(df[df['Tuberculosis'] == 0])

    if tb == 0 or no_tb == 0:
        print(f'Split {name} has a ratio of infnity, which is BAD')
        return False

    ratio = no_tb / tb

    if ratio > 0.9 and ratio < 1.2:
        return True
    else:
        print(f'Split {name} has a ratio of {ratio}, which is BAD')
        return False

def print_summary(df, name):
    total_len = len(df)
    no_finding = len(df[df['No Finding'] == 0])
    tb = len(df[df['Tuberculosis'] == 0])

    print(f'CSV: {name}, No Finding: {no_finding}, Tuberculosis: {tb}')


def perform_split(root_path, parsed):
    
    okay = False
    while not okay:
        val_rows = []
        test_rows = []
        train_rows = []
        
        try:
            for stuff in tqdm(parsed):
                rnd = random.random()

                if rnd < VAL_RATIO:
                    val_rows.append(stuff)
                elif rnd < VAL_RATIO + TEST_RATIO:
                    test_rows.append(stuff)
                else:
                    train_rows.append(stuff)

            val_df = pd.DataFrame(val_rows)
            assert verify_one_split(val_df, 'val', ratio=VAL_RATIO)
            val_df.to_csv(root_path / f'chexpert_like_val.csv')
            print_summary(val_df, 'validation')

            test_df = pd.DataFrame(test_rows)
            assert verify_one_split(test_df, 'test', ratio=TEST_RATIO)
            test_df.to_csv(root_path / f'chexpert_like_test.csv')
            print_summary(test_df, 'test')

            okay = True
        except AssertionError:
            pass
    
    ratios = ALL_SEMI_RATIO + [1]
    for s in ratios:
        for it in range(SEMI_ITERATIONS[s]):
            
            df = None
            name = f'{s}_{it}'
            while not verify_one_split(df, name):
                items = []
                for item in train_rows:
                    rnd = random.random()
                    if rnd < s:
                        items.append(item)

                df = pd.DataFrame(items)
                verify_one_split(df, name, s * TOTAL_TRAIN_RATIO)

            df.to_csv(root_path / f'chexpert_like_{name}.csv')
            print_summary(df, name)


def convert_shenzhen(root_folder):

    RE_SEX_AGE = re.compile(r'(?P<sex>.*al)[e]?[\s|,]*(?P<age>[0-9]+)[yr]?[s]?')
    RE_FNAME = re.compile(r'CHNCXR\_(?P<idx>[0-9]+)\_(?P<lbl>[0|1])\.txt')

    root_path = Path(root_folder)

    key_words = ['upper', 'lower', 'left', 'right', 'bilateral', 'atb', 'ptb', 'stb']

    # readings = {'healthy': [], 'disease': []}
    parsed = []
    for i, f in tqdm(enumerate(os.listdir(root_path / 'ClinicalReadings'))):

        f_result = RE_FNAME.search(f)
        pid = f_result.groupdict()['idx']
        lbl = f_result.groupdict()['lbl']

        data = {
            'Study': None,
            'Age': None,
            'Sex': None,
            'No Finding': None,
            'Tuberculosis': None, 
            'Path': None
        }

        disease = None
        with open(root_path / 'ClinicalReadings' / f, 'r') as txt:
            lines = txt.readlines()

            # if len(lines) > 3:
            #    import pdb; pdb.set_trace()

            for l in lines:
                result = RE_SEX_AGE.search(l)

                if result:
                    age = int(result.groupdict()['age'])
                    sex = result.groupdict()['sex'].lower()

                    data['Age'] = age
                    data['Sex'] = sex
                else:
                    l = l.strip().lower()

                    if len(l) > 0:
                        if 'normal' in l:
                            assert lbl == '0'
                            disease = False
                        else:
                            if lbl != '1':
                                import pdb; pdb.set_trace()

                            disease = False
                            for k in key_words:
                                if k in l:
                                    disease = True
                            
                            if 'pleuritis' in l:
                                disease = True

            assert disease is not None
            
        if disease:
            data['No Finding'] = 0
            data['Tuberculosis'] = 1
        else:
            data['No Finding'] = 1
            data['Tuberculosis'] = 0
        
    
        fname = root_path / 'shenzhentest' / 'test' / f'patient{pid}' / 'study1' / 'view1_frontal.jpg'
        study = Path('shenzhen') / 'shenzhentest' / 'test' / f'patient{pid}' / 'study1'
        data['Study'] = study
        data['Path'] = fname

        parsed.append(data)
    
    perform_split(root_path, parsed)

if __name__ == '__main__':
    # Usage:
    #   python shenzhen_mutiple_split.py moco/shenzhen
    # Try 17, 28, 20

    convert_shenzhen(sys.argv[1])