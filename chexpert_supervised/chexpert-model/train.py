"""Entry-point script to train models."""
import torch
import torch.nn as nn

import models
from args import TrainArgParser
from logger import Logger
from saver import ModelSaver
from predict import Predictor
from data import get_loader
from eval import Evaluator
from optim import Optimizer
from constants import *


def train(args):
    """Run model training."""

    print("Start Training ...")

    # Get nested namespaces.
    model_args = args.model_args
    logger_args = args.logger_args
    optim_args = args.optim_args
    data_args = args.data_args
    transform_args = args.transform_args


    # Get logger.
    print ('Getting logger... log to path: {}'.format(logger_args.log_path))
    logger = Logger(logger_args.log_path, logger_args.save_dir)

    # For conaug, point to the MOCO pretrained weights.
    if model_args.ckpt_path and model_args.ckpt_path != 'None':
        print("pretrained checkpoint specified : {}".format(model_args.ckpt_path))
        # CL-specified args are used to load the model, rather than the
        # ones saved to args.json.
        model_args.pretrained = False
        ckpt_path = model_args.ckpt_path
        model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                                 gpu_ids=args.gpu_ids,
                                                 model_args=model_args,
                                                 is_training=True)
        

        if not model_args.moco:
            optim_args.start_epoch = ckpt_info['epoch'] + 1
        else:
            optim_args.start_epoch = 1
    else:
        print('Starting without pretrained training checkpoint, random initialization.')
        # If no ckpt_path is provided, instantiate a new randomly
        # initialized model.
        model_fn = models.__dict__[model_args.model]
        if data_args.custom_tasks is not None:
            tasks = NamedTasks[data_args.custom_tasks]
        else:
            tasks = model_args.__dict__[TASKS]  # TASKS = "tasks"
        print("Tasks: {}".format(tasks))
        model = model_fn(tasks, model_args)
        model = nn.DataParallel(model, args.gpu_ids)


    # Put model on gpu or cpu and put into training mode.
    model = model.to(args.device)
    model.train()

    print("========= MODEL ==========")
    print(model)

    # Get train and valid loader objects.
    train_loader = get_loader(phase="train",
                             data_args=data_args,
                             transform_args=transform_args,
                             is_training=True,
                             return_info_dict=False,
                             logger=logger)
    valid_loader = get_loader(phase="valid",
                              data_args=data_args,
                              transform_args=transform_args,
                              is_training=False,
                              return_info_dict=False,
                              logger=logger)

    # Instantiate the predictor class for obtaining model predictions.
    predictor = Predictor(model, args.device)
    # Instantiate the evaluator class for evaluating models.
    evaluator = Evaluator(logger)
    # Get the set of tasks which will be used for saving models
    # and annealing learning rate.
    eval_tasks = EVAL_METRIC2TASKS[optim_args.metric_name]

    # Instantiate the saver class for saving model checkpoints.
    saver = ModelSaver(save_dir=logger_args.save_dir,
                       iters_per_save=logger_args.iters_per_save,
                       max_ckpts=logger_args.max_ckpts,
                       metric_name=optim_args.metric_name,
                       maximize_metric=optim_args.maximize_metric,
                       keep_topk=logger_args.keep_topk)

    # TODO: JBY: handle threshold for fine tuning
    if model_args.fine_tuning == 'full': # Fine tune all layers. 
        pass
    else:
        # Freeze other layers.
        models.PretrainedModel.set_require_grad_for_fine_tuning(model, model_args.fine_tuning.split(','))

    # Instantiate the optimizer class for guiding model training.
    optimizer = Optimizer(parameters=model.parameters(),
                          optim_args=optim_args,
                          batch_size=data_args.batch_size,
                          iters_per_print=logger_args.iters_per_print,
                          iters_per_visual=logger_args.iters_per_visual,
                          iters_per_eval=logger_args.iters_per_eval,
                          dataset_len=len(train_loader.dataset),
                          logger=logger)

    if model_args.ckpt_path and not model_args.moco:
        # Load the same optimizer as used in the original training.
        optimizer.load_optimizer(ckpt_path=model_args.ckpt_path,
                                 gpu_ids=args.gpu_ids)

    model_uncertainty = model_args.model_uncertainty
    loss_fn = evaluator.get_loss_fn(loss_fn_name=optim_args.loss_fn,
                                    model_uncertainty=model_args.model_uncertainty,
                                    mask_uncertain=True,
                                    device=args.device)

    # Run training
    while not optimizer.is_finished_training():
        optimizer.start_epoch()

        # TODO: JBY, HACK WARNING  # What is the hack?
        metrics = None
        for inputs, targets in train_loader:
            optimizer.start_iter()
            if optimizer.global_step and optimizer.global_step % optimizer.iters_per_eval == 0 or len(train_loader.dataset) - optimizer.iter < optimizer.batch_size:

                # Only evaluate every iters_per_eval examples.
                predictions, groundtruth = predictor.predict(valid_loader)
                # print("predictions: {}".format(predictions))
                metrics, curves = evaluator.evaluate_tasks(groundtruth, predictions)
                # Log metrics to stdout.
                logger.log_metrics(metrics)

                # Add logger for all the metrics for valid_loader
                logger.log_scalars(metrics, optimizer.global_step)

                # Get the metric used to save model checkpoints.
                average_metric = evaluator.evaluate_average_metric(metrics,
                                                      eval_tasks,
                                                      optim_args.metric_name)

                if optimizer.global_step % logger_args.iters_per_save == 0:
                    # Only save every iters_per_save examples directly
                    # after evaluation.
                    print("Save global step: {}".format(optimizer.global_step))
                    saver.save(iteration=optimizer.global_step,
                               epoch=optimizer.epoch,
                               model=model,
                               optimizer=optimizer,
                               device=args.device,
                               metric_val=average_metric)

                # Step learning rate scheduler.
                optimizer.step_scheduler(average_metric)

            with torch.set_grad_enabled(True):
                logits, embedding = model(inputs.to(args.device))
                loss = loss_fn(logits, targets.to(args.device))
                optimizer.log_iter(inputs, logits, targets, loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            optimizer.end_iter()

        optimizer.end_epoch(metrics)

    logger.log('=== Training Complete ===')

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParser()
    train(parser.parse_args())
