import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import os

import pandas as pd
import cv2
import torch
import numpy as np
from imageio import imsave

import util
from dataset import TASK_SEQUENCES
from cams import GradCAM, EnsembleCAM
from cams import GuidedBackPropagation
from saver import ModelSaver
from args import TestArgParser
from dataset import get_loader, get_eval_loaders
from dataset.constants import IMAGENET_MEAN, IMAGENET_STD

def save_grad_cams(args, loader, model, output_dir, only_competition=False, only_top_task=False):
    """Save grad cams for all examples in a loader."""

    # 'study_level' determined if the loader is returning
    # studies or individual images
    study_level = loader.dataset.study_level
    
    # NOTE: some model does not have task_sequence
    if hasattr(model.module, 'task_sequence'):
        task_sequence = model.module.task_sequence
    # NOTE: Right now hard code to "stanford" task_sequence,
    # to match the number of predictions CheXpert makes.
    else:
        # task_sequence = TASK_SEQUENCES[data_args.task_sequence]
        task_sequence = TASK_SEQUENCES["stanford"]
        print(f'WARNING: assuming that the models task sequence is \n {task_sequence}')

    if hasattr(model, "task2model_dicts"):
        grad_cam = EnsembleCAM(model, args.device)
    else:
        grad_cam = GradCAM(model, args.device)

    # By keeping track of the example id
    # we can name each folder using the example_id.
    counter = 0

    if study_level:
        # for inputs_batch, labels_batch, masks_batch in loader:
        for inputs_batch, labels_batch, info_batch, masks_batch in loader:
            for i, (input_study, label_study, mask_study) in enumerate(zip(inputs_batch, labels_batch, masks_batch)):

                directory = f'{output_dir}/{counter}'
                # Loop over the views in a studyo
                view_id = 0
                for input_, mask_val in zip(input_study, mask_study):
                    # Skip this image if it is just a 'padded' image
                    if mask_val == 0:
                        continue

                    write_grad_cams(input_, label_study, grad_cam, directory,
                                    task_sequence,
                                    only_competition=only_competition,
                                    view_id=view_id)
                    view_id = view_id + 1

                # Write label to txt and save to same folder
                # to make inspecting the cams easier
                label = np.reshape(label_study.numpy(), (1, -1))
                label_df = pd.DataFrame(label, columns=list(task_sequence))
                label_df["Path"] = info_batch['paths'][i]
                label_df["Counter"] = counter
                label_df.to_csv(f'{directory}/groundtruth.txt', index=False)

                counter = counter + 1

    else:
        for inputs, labels in loader:
            for input_, label in zip(inputs, labels):
                directory = f'{output_dir}/{counter}'
                write_grad_cams(input_, label, grad_cam, directory, task_sequence)

            counter = counter + 1

def write_grad_cams(input_, label, grad_cam,
        directory, task_sequence, only_competition=False, only_top_task=False, view_id=None):

    """Creates a CAM for each image.

        Args:
            input: Image tensor with shape (3 x h x h)
            grad_cam: EnsembleCam Object wrapped around GradCam objects, which are wrapped around models.
            directory: the output folder for these set of cams
            task_sequence:
    """
    if only_competition:
        COMPETITION_TASKS = TASK_SEQUENCES['competition']

    # Get the original image by
    # unnormalizing (img pixels will be between 0 and 1)
    # img shape: c, h, w
    img = util.un_normalize(input_, IMAGENET_MEAN, IMAGENET_STD)

    # move rgb chanel to last
    img = np.moveaxis(img, 0, 2)

    # Add the batch dimension
    # as the model requires it.
    input_ = input_.unsqueeze(0)
    _, channels, height, width = input_.shape
    num_tasks = len(task_sequence)

    # Create the directory for cams for this specific example
    if not os.path.exists(directory):
        os.makedirs(directory)

    #assert (inputs.shape[0] == 1), 'batch size must be equal to 1'
    with torch.set_grad_enabled(True):

        for task_id in range(num_tasks):
            task_name = list(task_sequence)[task_id]
            if only_competition:
                if task_name not in COMPETITION_TASKS:
                    continue

            task = task_name.lower()
            task = task.replace(' ', '_')
            task_label = int(label[task_id].item())
            if any([((task in f) and (f'v-{view_id}' in f)) for f in os.listdir(directory)]) or task_label != 1:
                continue

            probs, idx, cam = grad_cam.get_cam(input_, task_id, task_name)

            # Resize cam and overlay on image
            resized_cam = cv2.resize(cam, (height, width))
            # We don't normalize since the grad clam class has already taken care of that
            img_with_cam = util.add_heat_map(img, resized_cam, normalize=False)

            # Save a cam for this task and image
            # using task, prob and groundtruth in file name
            prob = probs[idx==task_id].item()
            if view_id is None:
                filename = f'{task}-p{prob:.3f}-gt{task_label}.png'
            else:
                filename = f'{task}-p{prob:.3f}-gt{task_label}-v-{view_id}.png'
            output_path = os.path.join(directory, filename)
            imsave(output_path, img_with_cam)


    # Save the original image in the same folder
    output_path = os.path.join(directory, f'original_image-v-{view_id}.png')
    img = np.uint8(img * 255)
    imsave(output_path, img)

