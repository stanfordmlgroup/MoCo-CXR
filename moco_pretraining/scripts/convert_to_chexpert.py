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
TEST_RATIO = 0.15
VAL_RATIO = 0.1


def print_summary(df, name):
    total_len = len(df)
    no_finding = len(df[df['No Finding'] == 0])
    tb = len(df[df['Tuberculosis'] == 0])

    print(f'CSV: {name}, No Finding: {no_finding}, Tuberculosis: {tb}')


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
    
    val_rows = []
    test_rows = []

    ratios = ALL_SEMI_RATIO + [1]
    fine_tune_splitted_rows = {s: [] for s in ratios}
    for stuff in tqdm(parsed):
        rnd = random.random()

        if rnd < VAL_RATIO:
            val_rows.append(stuff)
        elif rnd < VAL_RATIO + TEST_RATIO:
            test_rows.append(stuff)
        else:
            rnd = random.random()

            for s in ratios:
                if rnd < s:
                    fine_tune_splitted_rows[s].append(stuff)

    df = pd.DataFrame(val_rows)
    df.to_csv(root_path / f'chexpert_like_val.csv')
    print_summary(df, 'validation')

    df = pd.DataFrame(test_rows)
    df.to_csv(root_path / f'chexpert_like_test.csv')
    print_summary(df, 'test')

    for s in ratios:
        df = pd.DataFrame(fine_tune_splitted_rows[s])
        df.to_csv(root_path / f'chexpert_like_{s}.csv')
        print_summary(df, f'semi_{s}')

if __name__ == '__main__':
    # Usage:
    #   python convert_to_chexpert.py moco/shenzhen 23
    # Try 17, 28, 20

    random.seed(sys.argv[2])

    convert_shenzhen(sys.argv[1])