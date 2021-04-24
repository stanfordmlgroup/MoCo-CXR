# Create dummy csv and dummy image folders before running test
import subprocess
import shutil
import os
import glob
import csv
from constants import *
from argparse import ArgumentParser


def parse_script_args():
    """Parse command line arguments.

    Returns:
        args (Namespace): Parsed command line arguments

    """
    parser = ArgumentParser()

    parser.add_argument('--save_dir',
                        type=str, default=str(CHEXPERT_SAVE_DIR),
                        help='Directory to save model data.')

    parser.add_argument('--img_folder', type=str,
                        default=None, required=True,
                        help='Path to folder of all the images')

    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size for training / evaluation.')

    parser.add_argument('--ckpt_path',
                        type=str, default=None,
                        help=('Checkpoint path for eval.'))
    
    parser.add_argument('--config_path',
                        type=str, default=None,
                        help=('Path to ensemble.'))

    args = parser.parse_args()
    return args

def folders_csv(folder):
    """Create csv and put images in folder
    
    Args:
        folder (str): path to all the images
    """
    images = glob.glob(folder + "/*.jpg")
    rows = []
    for image in images:
        img_path = Path(image)
        img_name = img_path.name
        new_dir = img_path.parent / img_name.rstrip('.jpg')
        new_dir.mkdir(exist_ok=True, parents=True)
        new_path = new_dir / img_name
        rows.append([str(new_path.absolute())] + [None] * 4 + [0] * 14)
        img_path.rename(new_path)
    with open(folder + '/dummy.csv', 'w') as csv_file:
        row = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] \
                + CHEXPERT_TASKS
        writer = csv.writer(csv_file)
        writer.writerow(row)
        for row in rows:
            writer.writerow(row)

def run_test(args):
    """Run test on dummy csv

    Args:
        args (Namespace): Parsed command line arguments
    """
    if args.config_path is not None:
        path = "--config_path"
        path_name = args.config_path
    else:
        path = "--ckpt_path"
        path_name = args.ckpt_path
    subprocess.run(['python', 'test.py', '--dataset', 'custom', path, path_name,
                    '--phase', 'test', '--together', 'True', '--test_csv',
                    str(args.img_folder + '/dummy.csv'), '--save_dir', args.save_dir])
    os.remove(args.img_folder + '/dummy.csv')     #remove if you want to keep csv
    os.remove(args.save_dir + '/results/test/groundtruth.csv')


if __name__ == "__main__":
    args = parse_script_args()
    csv = folders_csv(args.img_folder)
    run_test(args)