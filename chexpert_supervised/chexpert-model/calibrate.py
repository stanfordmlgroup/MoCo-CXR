"""Entry-point script to train models."""
import torch

from args import TestArgParser
from logger import Logger
from predict import Predictor, EnsemblePredictor
from saver import ModelSaver
from data import get_loader
from eval import Evaluator
from constants import *


def calibrate(args):
    """Run model testing."""
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args

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
        # Obtain model predictions
        if return_info_dict:
            predictions, groundtruth, paths = predictor.predict(loader)
        else:
            predictions, groundtruth = predictor.predict(loader)
        #print(groundtruth)
    # custom function
    from sklearn.linear_model import LogisticRegression as LR
    params = []
    for column in predictions:
        #print(predictions[column].values)
        #print(groundtruth[column].values)
        #drop corresponding rows where gt is -1  and 
        lr = LR(C=15)
        to_drop = groundtruth.index[groundtruth[column] == -1].tolist()                                        
        lr.fit(predictions[column].drop(to_drop).values.reshape(-1,1),groundtruth[column].drop(to_drop).values)     # LR needs X to be 2-dimensional
        print("num_rows_used",predictions[column].drop(to_drop).values.size)
        #print(groundtruth[column].drop(to_drop).values.size)
        #print(predictions[column].values)
        print("coeffs", lr.coef_, lr.intercept_)
        p_calibrated=lr.predict_proba(predictions[column].values.reshape(-1,1))
        params.append((lr.coef_, lr.intercept_))
    import json
    with open('calibration_params.json', 'w') as f:
        import pandas as pd
        pd.Series(params).to_json(f, orient='values')
 
    #return lr

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    calibrate(parser.parse_args())
     
