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

random.seed(0)

# ALL_SEMI_RATIO =  [0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]
ALL_SEMI_RATIO =  [0.001, 0.01, 0.1, 0.5]

SEMI_ITERATIONS = { 0.001: 5,
                    0.01: 5,
                    0.1: 5,
                    0.5: 5,
                    1: 1
                }
print(ALL_SEMI_RATIO)


def move_montgomery(root_folder, truth_csv, destination_root):

    root_path = Path(root_folder)
    os.makedirs(destination_root, exist_ok=True)

    truth = pd.read_csv(truth_csv)

    healthy = truth[truth['No Finding'] == 1]
    disease = truth[truth['Consolidation'] == 1]

    dst_path = Path(destination_root) / 'healthy'
    os.makedirs(str(dst_path), exist_ok=True)
    for i, row in healthy.iterrows():
        fname = row['Path']
        path = fname.split('/')[-4:-1]

        fname = fname.replace('view.jpg', 'view1_frontal.jpg')
        path.append('view1_frontal.jpg')

        dst_fname = dst_path / f'{path[-3]}.jpg'

        # print (str(root_path / fname))
        shutil.copy(str(root_path / fname), dst_fname)
        # print(dst_fname)
    
    dst_path = Path(destination_root) / 'disease'
    os.makedirs(str(dst_path), exist_ok=True)
    for i, row in disease.iterrows():
        fname = row['Path']
        path = fname.split('/')[-4:-1]

        fname = fname.replace('view.jpg', 'view1_frontal.jpg')
        path.append('view1_frontal.jpg')

        dst_fname = dst_path / f'{path[-3]}.jpg'

        # print (str(root_path / fname))
        shutil.copy(str(root_path / fname), dst_fname)
        # print(dst_fname)


def move_shenzhen(root_folder, destination_root):

    RE_SEX_AGE = re.compile(r'(?P<sex>.*al)[e]?[\s|,]*(?P<age>[0-9]+)[yr]?[s]?')
    RE_FNAME = re.compile(r'CHNCXR\_(?P<idx>[0-9]+)\_(?P<lbl>[0|1])\.txt')

    root_path = Path(root_folder)
    os.makedirs(destination_root, exist_ok=True)

    key_words = ['upper', 'lower', 'left', 'right', 'bilateral', 'atb', 'ptb', 'stb']

    # readings = {'healthy': [], 'disease': []}
    parsed = []
    for i, f in enumerate(os.listdir(root_path / 'ClinicalReadings')):
        
        f_result = RE_FNAME.search(f)
        pid = f_result.groupdict()['idx']
        lbl = f_result.groupdict()['lbl']

        with open(root_path / 'ClinicalReadings' / f, 'r') as txt:
            lines = txt.readlines()

            # if len(lines) > 3:
            #    import pdb; pdb.set_trace()

            for l in lines:
                result = RE_SEX_AGE.search(l)

                if result:
                    age = int(result.groupdict()['age'])
                    sex = result.groupdict()['sex'].lower()

                    data = {'age': age, 'sex': sex, 'index': i, 'patient': pid, 'healthy': 1, 'fname': f.split('.')[0]}
                    symp = {k: 0 for k in key_words}
                    data.update(symp)

                else:
                    l = l.strip().lower()

                    if len(l) > 0:
                        if 'normal' in l:
                            assert lbl == '0'
                        else:
                            if lbl != '1':
                                import pdb; pdb.set_trace()

                            data['healthy'] = 0
                            for k in key_words:
                                if k in l:
                                    data[k] = 1
                            
                            if 'pleuritis' in l:
                                data['PTB'] = 1

            
            parsed.append(data)
    
    df = pd.DataFrame.from_dict(parsed)

    healthy = df[df['healthy'] == 1]
    disease = df[df['healthy'] == 0]

    # import pdb; pdb.set_trace()

    dst_path = Path(destination_root) / 'healthy'
    os.makedirs(str(dst_path), exist_ok=True)
    for i, row in tqdm(healthy.iterrows()):
        fname = root_path / 'CXR_png' / f"{row['fname']}.png"

        dst_fname = dst_path / f"{row['fname']}.png"

        # print (str(root_path / fname))
        shutil.copy(str(root_path / fname), dst_fname)
        # print(dst_fname)
    
    dst_path = Path(destination_root) / 'disease'
    os.makedirs(str(dst_path), exist_ok=True)
    for i, row in tqdm(disease.iterrows()):
        fname = root_path / 'CXR_png' / f"{row['fname']}.png"

        dst_fname = dst_path / f"{row['fname']}.png"

        # print (str(root_path / fname))
        shutil.copy(str(root_path / fname), dst_fname)
        # print(dst_fname)


def move_chexpert_single_target(root_folder, destination_root, disease_name):
    '''
    root_folder is /deep/group/CheXpert/CheXpert-v1.0-small
    destination_root is wherever we move data to, all things will be symlink
    disease is things like "Pleural Effusion"

    Uncertain labels (-1) are POSITIVE (1)
    '''

    # Data is split into
    # valid (serve as actual test, these are hand labeled)
    # From original training set
    #   actual_val (serve as actual validation data, not test)
    #   For each fine tune ratio, we split the remaining into 2 more folders
    #       actual_train (serve as unlabeled training)
    #       fine_tune

    actual_val_ratio = 0.1
    # fine_tune_ratio_list = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]
    fine_tune_ratio_list = ALL_SEMI_RATIO

    root_path = Path(root_folder)

    def move_to_category(category, folder, df):
        dst_path = Path(destination_root) / 'data' / folder / category
        os.makedirs(str(dst_path), exist_ok=True)

        df_path = Path(destination_root) / 'moving_logs' / f'{folder}_log.csv'
        # TODO: JBY The splitting part is a bit of a hack, but welp, too lazy to fix
        os.makedirs(str(Path(destination_root) / 'moving_logs' / folder.split('/')[0]), exist_ok=True)
        df.to_csv(df_path)

        new_paths = []
        for i, row in tqdm(df.iterrows()):
            fname = row['Path']
            splitted_fname = fname.split('/')
            new_fname = '_'.join(splitted_fname[-3:])

            dst_fname = dst_path / new_fname

            desired_path = Path(root_path / '/'.join(splitted_fname[1:]))
            if not dst_fname.exists():
                os.symlink(str(desired_path), dst_fname)
            else:
                # Do nothing for now
                pass
                
            new_paths.append(dst_fname)

        return new_paths

    def split_df(df, disease_name):
        '''Split a ground truth dataframe into no symptom and disease,
            Uncertain labels are treated as disease

            Note that we are NOT handling "No Finding" but only wrt a disease
        '''
        # healthy = df[df['No Finding'] == 1]
        no_sym = df[(df[disease_name] == 0) | (df[disease_name].isnull().values)]
        disease = df[(df[disease_name] == 1) | (df[disease_name] == -1)]
        print(f'Original: {str(len(df)).ljust(8)}\t\tNo Symptom: {len(no_sym)}, {disease_name}: {len(disease)}')

        assert len(df) == len(no_sym) + len(disease)
        return no_sym, disease

    
    os.makedirs(destination_root, exist_ok=True)
    disease_short_name = disease_name.replace(' ', '_').lower()

    ### Move original valid into test ####
    print('===== Moving valid into test =====')
    truth_csv = root_path / f'valid.csv'
    truth_df = pd.read_csv(truth_csv)
        
    no_sym, disease = split_df(truth_df, disease_name)
    move_to_category('no_sym', 'test', no_sym)
    move_to_category(disease_short_name, 'test', disease)


    ### Now handle splitting train into validation, sub folders of actual training
    ### with proper ratio for fine tuning
    truth_csv = root_path / f'train.csv'
    truth_df = pd.read_csv(truth_csv)

    RE_PARSE = re.compile(r'.*patient(?P<patient>[0-9]*)\/study(?P<study>[0-9]*)\/(?P<img>.*)\.jpg')

    print('===== Organizing by patient =====')
    structure = defaultdict(lambda: defaultdict(list))
    for i, row in truth_df.iterrows():

        result = RE_PARSE.search(row['Path'])
        patient = result.groupdict()['patient']
        study = result.groupdict()['study']
        img = result.groupdict()['img']

        structure[patient][study].append((i, img))

    actual_valid_ilocs = []
    fine_tune_train_ilocs = []
    fine_tune_tune_ilocs = [[] for i in range(len(fine_tune_ratio_list))]

    def assign_all_of_patient(structure, patient, target_list):
        for study in structure[patient]:
            for iloc, img in structure[patient][study]:
                target_list.append(iloc)
        return target_list

    print('===== Splitting for validation and fine tuning =====')
    for patient in structure:
        if random.random() < actual_val_ratio:
            assign_all_of_patient(structure, patient, actual_valid_ilocs)
        else:
            for i, fine_tune_ratio in enumerate(fine_tune_ratio_list):
                if random.random() < fine_tune_ratio:
                    fine_tune_tune_ilocs[i] = assign_all_of_patient(structure, patient, fine_tune_tune_ilocs[i])
                # else:
            fine_tune_train_ilocs = assign_all_of_patient(structure, patient, fine_tune_train_ilocs)

    print('===== Moving validation =====')
    actual_valid_df = truth_df.iloc[actual_valid_ilocs]
    av_healthy, av_disease = split_df(actual_valid_df, disease_name)
    move_to_category('no_sym', 'valid', av_healthy)
    move_to_category(disease_short_name, 'valid', av_disease)

    print ('===== Moving training for fine tuning =====')
    fine_tune_train_df = truth_df.iloc[fine_tune_train_ilocs]
    tr_healthy, tr_disease = split_df(fine_tune_train_df, disease_name)
    move_to_category('no_sym', f'fine_tune_train', tr_healthy)
    move_to_category(disease_short_name, f'fine_tune_train', tr_disease)

    for i, fine_tune_ratio in enumerate(fine_tune_ratio_list):
        print(f'===== Moving data for ratio={fine_tune_ratio} =====')
        # os.makedirs(destination_root + '/' + f'semi_{fine_tune_ratio}', exist_ok=True)

        fine_tune_tune_df = truth_df.iloc[fine_tune_tune_ilocs[i]]
        ft_healthy, ft_disease = split_df(fine_tune_tune_df, disease_name)
        move_to_category('no_sym', f'semi_{fine_tune_ratio}', ft_healthy)
        move_to_category(disease_short_name, f'semi_{fine_tune_ratio}', ft_disease)

        # We do not have to actually copy the training set for fine tuning
        #fine_tune_train_df = truth_df.iloc[fine_tune_train_ilocs[i]]
        #tr_healthy, tr_disease = split_df(fine_tune_train_df, disease_name)
        #move_to_category('no_sym', f'semi_{fine_tune_ratio}/train', tr_healthy)
        #move_to_category(disease_short_name, f'semi_{fine_tune_ratio}/train', tr_disease)


def move_chexpert_codalab_test(root_folder, destination_root, disease_name):
    '''
    root_folder is /deep/group/CheXpert/CheXpert-v1.0-small
    test_folder is /deep/group/CheXpert/CodaLab
    destination_root is wherever we move data to, all things will be symlink
    disease is things like "Pleural Effusion"

    Uncertain labels (-1) are POSITIVE (1)
    '''

    # Data is split into
    # valid (serve as actual test, these are hand labeled)
    # From original training set
    #   actual_val (serve as actual validation data, not test)
    #   For each fine tune ratio, we split the remaining into 2 more folders
    #       actual_train (serve as unlabeled training)
    #       fine_tune

    actual_val_ratio = 0.1
    fine_tune_ratio_list = ALL_SEMI_RATIO
    print(f'Fine tuning ratios are {fine_tune_ratio_list}')

    root_path = Path(root_folder)
    codalab_path = Path('/deep/group/CheXpert/CodaLab')

    def move_to_category(category, folder, df, source=root_path):
        dst_path = Path(destination_root) / 'data' / folder / category
        os.makedirs(str(dst_path), exist_ok=True)

        df_path = Path(destination_root) / 'moving_logs' / f'{folder}_{category}_log.csv'
        # TODO: JBY The splitting part is a bit of a hack, but welp, too lazy to fix
        os.makedirs(str(Path(destination_root) / 'moving_logs' / folder.split('/')[0]), exist_ok=True)

        new_paths = []
        for i, row in tqdm(df.iterrows()):
            fname = row['Path']
            splitted_fname = fname.split('/')
            new_fname = '_'.join(splitted_fname[-3:])

            dst_fname = dst_path / new_fname

            desired_path = Path(source / '/'.join(splitted_fname[1:]))
            if not dst_fname.exists():
                os.symlink(str(desired_path), dst_fname)
            else:
                # Do nothing for now
                pass
        
            new_paths.append(desired_path)

        df['Path'] = new_paths
        df.to_csv(df_path)


    def split_df(df, disease_name):
        '''Split a ground truth dataframe into no symptom and disease,
            Uncertain labels are treated as disease

            Note that we are NOT handling "No Finding" but only wrt a disease
        '''
        # healthy = df[df['No Finding'] == 1]
        no_sym = df[(df[disease_name] == 0) | (df[disease_name].isnull().values)]
        disease = df[(df[disease_name] == 1) | (df[disease_name] == -1)]
        print(f'Original: {str(len(df)).ljust(8)}\t\tNo Symptom: {len(no_sym)}, {disease_name}: {len(disease)}')

        assert len(df) == len(no_sym) + len(disease)
        return no_sym, disease

    def join_gt_fnames(gt, fname_path):
        '''
        we want fnames is of format {path: [list of files] }
        '''

        fnames = defaultdict(list)
        with open(fname_path, 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                l = l.strip()
                study = '/'.join(l.split('/')[:-1])
                fnames[study].append(l)

        result_list = []
        for i, row in gt.iterrows():
            for f in fnames[row['Study']]:
                cur_row = copy.deepcopy(row)
                cur_row['Path'] = f
                result_list.append(cur_row)
        
        result_df = pd.DataFrame(result_list)
        return result_df
    
    os.makedirs(destination_root, exist_ok=True)
    disease_short_name = disease_name.replace(' ', '_').lower()

    
    ### Move original valid into test ####
    print('===== Moving CodaLab into test =====')
    truth_csv = codalab_path / f'test_groundtruth.csv'
    truth_df = pd.read_csv(truth_csv)
    truth_df = join_gt_fnames(truth_df, codalab_path / 'test_image_paths.csv')
        
    no_sym, disease = split_df(truth_df, disease_name)
    move_to_category('no_sym', 'test', no_sym, source=codalab_path)
    move_to_category(disease_short_name, 'test', disease, source=codalab_path)


    ### Move original valid into test ####
    print('===== Moving valid into valid =====')
    truth_csv = root_path / f'valid.csv'
    truth_df = pd.read_csv(truth_csv)
        
    no_sym, disease = split_df(truth_df, disease_name)
    move_to_category('no_sym', 'valid', no_sym)
    move_to_category(disease_short_name, 'valid', disease)


    ### Now handle splitting train into validation, sub folders of actual training
    ### with proper ratio for fine tuning
    truth_csv = root_path / f'train.csv'
    truth_df = pd.read_csv(truth_csv)

    RE_PARSE = re.compile(r'.*patient(?P<patient>[0-9]*)\/study(?P<study>[0-9]*)\/(?P<img>.*)\.jpg')

    print('===== Organizing by patient =====')
    structure = defaultdict(lambda: defaultdict(list))
    for i, row in truth_df.iterrows():

        result = RE_PARSE.search(row['Path'])
        patient = result.groupdict()['patient']
        study = result.groupdict()['study']
        img = result.groupdict()['img']

        structure[patient][study].append((i, img))

    actual_valid_ilocs = []
    fine_tune_train_ilocs = []
    fine_tune_tune_ilocs = [[] for i in range(len(fine_tune_ratio_list))]

    def assign_all_of_patient(structure, patient, target_list):
        for study in structure[patient]:
            for iloc, img in structure[patient][study]:
                target_list.append(iloc)
        return target_list

    print('===== Splitting for fine tuning =====')
    for patient in structure:
        for i, fine_tune_ratio in enumerate(fine_tune_ratio_list):
            if random.random() < fine_tune_ratio:
                fine_tune_tune_ilocs[i] = assign_all_of_patient(structure, patient, fine_tune_tune_ilocs[i])
            # else:
        fine_tune_train_ilocs = assign_all_of_patient(structure, patient, fine_tune_train_ilocs)

    print ('===== Moving training for fine tuning =====')
    fine_tune_train_df = truth_df.iloc[fine_tune_train_ilocs]
    tr_healthy, tr_disease = split_df(fine_tune_train_df, disease_name)
    move_to_category('no_sym', f'full_train', tr_healthy)
    move_to_category(disease_short_name, f'full_train', tr_disease)

    for i, fine_tune_ratio in enumerate(fine_tune_ratio_list):
        print(f'===== Moving data for ratio={fine_tune_ratio} =====')
        # os.makedirs(destination_root + '/' + f'semi_{fine_tune_ratio}', exist_ok=True)

        fine_tune_tune_df = truth_df.iloc[fine_tune_tune_ilocs[i]]
        ft_healthy, ft_disease = split_df(fine_tune_tune_df, disease_name)
        move_to_category('no_sym', f'semi_{fine_tune_ratio}', ft_healthy)
        move_to_category(disease_short_name, f'semi_{fine_tune_ratio}', ft_disease)

        # We do not have to actually copy the training set for fine tuning
        #fine_tune_train_df = truth_df.iloc[fine_tune_train_ilocs[i]]
        #tr_healthy, tr_disease = split_df(fine_tune_train_df, disease_name)
        #move_to_category('no_sym', f'semi_{fine_tune_ratio}/train', tr_healthy)
        #move_to_category(disease_short_name, f'semi_{fine_tune_ratio}/train', tr_disease)


def move_chexpert_random(root_folder, destination_root):
    '''
    root_folder is /deep/group/CheXpert/CheXpert-v1.0-small (train.csv)
    validation data is also in /deep/group/CheXpert/CheXpert-v1.0-small (valid.csv)
    test_folder is /deep/group/CheXpert/CodaLab

    destination_root is wherever we move data to, all things will be symlink
    '''

    # Data is split into
    # valid (serve as actual test, these are hand labeled)
    # From original training set
    #   actual_val (serve as actual validation data, not test)
    #   For each fine tune ratio, we split the remaining into 2 more folders
    #       actual_train (serve as unlabeled training)
    #       fine_tune

    fine_tune_ratio_list = ALL_SEMI_RATIO
    print(f'Fine tuning ratios are {fine_tune_ratio_list}')

    root_path = Path(root_folder)
    codalab_path = Path('/deep/group/CheXpert/CodaLab')

    def move_selected_data(df, source, destination_name):
        dst_path = Path(destination_root) / 'data' / destination_name
        os.makedirs(str(dst_path), exist_ok=True)

        df_path = Path(destination_root) / 'moving_logs' / f'{destination_name}.csv'
        os.makedirs(str(Path(destination_root) / 'moving_logs'), exist_ok=True)

        new_paths = []
        for i, row in tqdm(df.iterrows()):
            fname = row['Path']
            splitted_fname = str(fname).split('/')
            new_fname = '_'.join(splitted_fname[-3:])

            dst_fname = dst_path / new_fname

            desired_path = Path(source) / '/'.join(splitted_fname[1:])
            if not dst_fname.exists():
                os.symlink(str(desired_path), dst_fname)
            else:
                # Do nothing for now
                pass
        
            new_paths.append(desired_path)

        df['Path'] = new_paths
        df.to_csv(df_path)

    def join_gt_fnames(gt, fname_path):
        '''
        we want fnames is of format {path: [list of files] }
        '''

        fnames = defaultdict(list)
        with open(fname_path, 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                l = l.strip()
                study = '/'.join(l.split('/')[:-1])
                fnames[study].append(l)

        result_list = []
        for i, row in gt.iterrows():
            for f in fnames[row['Study']]:
                cur_row = copy.deepcopy(row)
                cur_row['Path'] = f
                result_list.append(cur_row)
        
        result_df = pd.DataFrame(result_list)
        return result_df
    
    os.makedirs(destination_root, exist_ok=True)
    
    ### Move original valid into test ####
    print('===== Moving CodaLab into test =====')
    truth_csv = codalab_path / f'test_groundtruth.csv'
    truth_df = pd.read_csv(truth_csv)
    truth_df = join_gt_fnames(truth_df, codalab_path / 'test_image_paths.csv')
        
    move_selected_data(truth_df, codalab_path, 'test')

    ### Move original valid into test ####
    print('===== Moving valid into valid =====')
    truth_csv = root_path / f'valid.csv'
    truth_df = pd.read_csv(truth_csv)
        
    move_selected_data(truth_df, root_folder, 'valid')

    ### Now handle splitting train into validation, sub folders of actual training
    ### with proper ratio for fine tuning
    print('===== Handling splitting of training data into smaller sets =====')
    truth_csv = root_path / f'train.csv'
    truth_df = pd.read_csv(truth_csv)

    # This way we ensure that lower semi-supervised ratios are subset of the higher ratio ones
    ratios = ALL_SEMI_RATIO + [1]
    fine_tune_splitted_rows = {s: [[] for it in range(SEMI_ITERATIONS[s])] for s in ratios}
    for stuff in tqdm(truth_df.iterrows()):
        i, row = stuff

        for s in ratios:
            for it in range(SEMI_ITERATIONS[s]):
                rnd = random.random()
                if rnd < s:
                    fine_tune_splitted_rows[s][it].append(row)

    for s in fine_tune_splitted_rows:
        for it in range(SEMI_ITERATIONS[s]):
            print(f'Ratio {s}, Iter {it}: {len(fine_tune_splitted_rows[s][it])}')

    print('===== Moving training data =====')
    for s in fine_tune_splitted_rows:
        for it in range(SEMI_ITERATIONS[s]):
            print(f'Processing {s}it{it}')
            splitted_df = pd.DataFrame(fine_tune_splitted_rows[s][it])

            if s == 1:
                move_selected_data(splitted_df, root_folder, f'train')
            else:
                move_selected_data(splitted_df, root_folder, f'train_semi_{s}-it{it}')


def combine_csvs(moving_folder):

    actual_name_map = {'full_train': 'train',
                       'valid': 'valid',
                       'test': 'test',
                       }
    actual_name_map.update({f'semi_{r}': f'train_semi_{r}' for r in ALL_SEMI_RATIO})

    all_division = ['full_train', 'valid', 'test'] + [f'semi_{r}' for r in ALL_SEMI_RATIO]
    labels  = ['no_sym', 'pleural_effusion']
    for division in all_division:
        all_df = []
        for flabel in labels:
            full_name = f'{moving_folder}/{division}_{flabel}_log.csv'
            df = pd.read_csv(full_name)
            all_df.append(df)
        
        joint = pd.concat(all_df)

        actual_fname = actual_name_map[division]
        full_name = f'{moving_folder}/{actual_fname}.csv'

        print(f'Produced {full_name}')
        joint.to_csv(full_name, index=False)


if __name__ == '__main__':

    if sys.argv[1] == 'montgomery':
        # root_folder, truth_csv, destination_root
        move_montgomery(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'shenzhen':
        # root_folder, destination_root
        move_shenzhen(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'chexpert_single':
        # root_folder, destination_root, disease name
        move_chexpert_single_target(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'chexpert_coda':
        '''
        root_folder, destination_root, disease name
        Example Usage:
            python reorganize_files.py chexpert_coda /deep/group/CheXpert/CheXpert-v1.0-small /deep/group/data/moco/chexpert-proper-test-2 "Pleural Effusion"
        '''
        move_chexpert_codalab_test(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'chexpert_random':
        '''
        root_folder, destination_root, disease name
        Example Usage:
            python reorganize_files.py chexpert_random /deep/group/CheXpert/CheXpert-v1.0-small /deep/group/data/moco/chexpert-proper-test-random-v2
        '''
        move_chexpert_random(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'join_logs':
        '''
        destination_folder
        Example Usage:
            python reorganize_files.py join_logs /deep/group/data/moco/chexpert-proper-test-2/moving_logs
        '''
        combine_csvs(sys.argv[2])
    else:
        print(f'Arguments {sys.argv} combination is not valid.')

