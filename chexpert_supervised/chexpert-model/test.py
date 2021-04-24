"""Entry-point script to train models."""
import torch

from args import TestArgParser
from logger import Logger
from predict import Predictor, EnsemblePredictor
from saver import ModelSaver
from data import get_loader
from eval import Evaluator
from constants import *
from scripts.get_cams import save_grad_cams
from dataset import TASK_SEQUENCES


def test(args):
    """Run model testing."""

    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args

    # import pdb; pdb.set_trace()

    # Get logger.
    logger = Logger(logger_args.log_path,
                    logger_args.save_dir,
                    logger_args.results_dir)

    # Get image paths corresponding to predictions for logging
    paths = None

    if model_args.config_path is not None:
        # Instantiate the EnsemblePredictor class for obtaining
        # model predictions.
        predictor = EnsemblePredictor(config_path=model_args.config_path,
                                      model_args=model_args,
                                      data_args=data_args,
                                      gpu_ids=args.gpu_ids,
                                      device=args.device,
                                      logger=logger)
        # Obtain ensemble predictions.
        # Caches both individual and ensemble predictions.
        # We always turn off caching to ensure that we write the Path column.
        predictions, groundtruth, paths = predictor.predict(cache=False,
                                                            return_paths=True,
                                                            all_gt_tasks=True)
    else:
        # Load the model at ckpt_path.
        ckpt_path = model_args.ckpt_path
        ckpt_save_dir = Path(ckpt_path).parent
        model_uncertainty = model_args.model_uncertainty
        # Get model args from checkpoint and add them to
        # command-line specified model args.
        model_args, transform_args\
            = ModelSaver.get_args(cl_model_args=model_args,
                                  dataset=data_args.dataset,
                                  ckpt_save_dir=ckpt_save_dir,
                                  model_uncertainty=model_uncertainty)
        
        # TODO JBY: in test moco should never be true.
        model_args.moco = args.model_args.moco
        model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                                 gpu_ids=args.gpu_ids,
                                                 model_args=model_args,
                                                 is_training=False)

        # Instantiate the Predictor class for obtaining model predictions.
        predictor = Predictor(model=model, device=args.device)
        # Get phase loader object.
        return_info_dict = True
        loader = get_loader(phase=data_args.phase,
                            data_args=data_args,
                            transform_args=transform_args,
                            is_training=False,
                            return_info_dict=return_info_dict,
                            logger=logger)
        # Obtain model predictions.
        if return_info_dict:
            predictions, groundtruth, paths = predictor.predict(loader)
        else:
            predictions, groundtruth = predictor.predict(loader)
        # print(predictions[CHEXPERT_COMPETITION_TASKS])
        if model_args.calibrate:
            #open the json file which has the saved parameters
            import json
            with open(CALIBRATION_FILE) as f:
                data = json.load(f)
            i = 0
            #print(predictions)
            import math
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))

            for column in predictions:
                predictions[column] = predictions[column].apply \
                                      (lambda x: sigmoid(x * data[i][0][0][0] \
                                      + data[i][1][0]))
                i += 1
        
            # print(predictions[CHEXPERT_COMPETITION_TASKS])
            #run forward on all the predictions in each row of predictions

    # Log predictions and groundtruth to file in CSV format.
    logger.log_predictions_groundtruth(predictions, groundtruth, paths)

    if not args.inference_only:
        # Instantiate the evaluator class for evaluating models.
        evaluator = Evaluator(logger,
                              operating_points_path=CHEXPERT_RAD_PATH)
        # Get model metrics and curves on the phase dataset.
        metrics, curves = evaluator.evaluate_tasks(groundtruth, predictions)
        # Log metrics to stdout and file.
        logger.log_stdout(f"Writing metrics to {logger.metrics_path}.")
        logger.log_metrics(metrics, save_csv=True)

    # TODO: make this work with ensemble
    # TODO: investigate if the eval_loader can just be the normal loader here
    if logger_args.save_cams:
        cams_dir = logger_args.save_dir / 'cams'
        print(f'Save cams to {cams_dir}')
        save_grad_cams(args, loader, model,
                       cams_dir,
                       only_competition=logger_args.only_competition_cams,
                       only_top_task=False)

    logger.log("=== Testing Complete ===")
    # Produce other visuals
    # TODO: This causes "unexpected error to scripts"
    # raise NotImplementedError()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    print("Start test...")
    test(parser.parse_args())
