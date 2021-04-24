import os 
import sys
import shutil
import random
import pandas as pd

random.seed(2020)

TRAIN_RATIO = 0.7


def split_folder(source_folder, target_train, target_val):

    os.makedirs(target_train, exist_ok=True)
    os.makedirs(target_val, exist_ok=True)

    for label in os.listdir(source_folder):
        os.makedirs(os.path.join(target_train, label), exist_ok=True)
        os.makedirs(os.path.join(target_val, label), exist_ok=True)

    allocation = []
    for label in os.listdir(source_folder):
        if os.path.isfile(os.path.join(source_folder, label)):
            continue
        for fname in os.listdir(os.path.join(source_folder, label)):

            source = os.path.join(source_folder, label, fname)
            train_path = os.path.join(target_train, label, fname)
            val_path = os.path.join(target_val, label, fname)

            if random.random() < TRAIN_RATIO:
                shutil.copy(source, train_path)
                # all_train[label].append(train_path)
                allocation.append({'orig_path': source, 'new_path': train_path,
                                   'train': 1, 'val': 0})
            else:
                shutil.copy(source, val_path)
                # all_val[label].append(val_path)
                allocation.append({'orig_path': source,  'new_path': val_path,
                                   'train': 0, 'val': 1})

    df = pd.DataFrame(allocation)
    df.to_csv(os.path.join(source_folder, 'assignment.csv'))


if __name__ == '__main__':

    source = sys.argv[1]
    train = sys.argv[2]
    val = sys.argv[3]

    split_folder(source, train, val)


    