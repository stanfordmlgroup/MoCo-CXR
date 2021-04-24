"""Define class for processing testing command-line arguments."""
import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--inference_only',
                                 action='store_true',
                                 help=('If set, then only do inference. Useful'+
                                       ' when the csv has uncertainty label'))
        # Data args
        self.parser.add_argument('--phase',
                                 dest='data_args.phase',
                                 type=str, default='valid',
                                 choices=('train', 'valid', 'test'))
        self.parser.add_argument('--test_groundtruth',
                                 dest='data_args.gt_csv',
                                 type=str, default=None,
                                 help=('csv file if custom dataset'))
        self.parser.add_argument('--test_image_paths',
                                 dest='data_args.paths_csv',
                                 type=str, default=None,
                                 help=('csv file if custom dataset'))
        self.parser.add_argument('--together',
                                 dest='data_args.together',
                                 type=str, default=True,
                                 help=('whether we have integrated test csv'))
        self.parser.add_argument('--test_csv',
                                 dest='data_args.test_csv',
                                 type=str, default=None,
                                 help=('csv file for integrated test set'))
        # Logger args
        self.parser.add_argument('--save_cams',
                                 dest='logger_args.save_cams',
                                 type=util.str_to_bool, default=False,
                                 help=('If true, will save cams to ' +
                                       'experiment_folder/cams'))
        self.parser.add_argument('--only_evaluation_cams',
                                 dest='logger_args.only_evaluation_cams',
                                 type=util.str_to_bool, default=True,
                                 help=('If true, will only generate cams ' +
                                       'on evaluation labels. Only ' +
                                       'relevant if --save_cams is True'))
        self.parser.add_argument('--only_competition_cams',
                                 dest='logger_args.only_competition_cams',
                                 type=util.str_to_bool, default=False,
                                 help='Whether to only output cams for' +
                                 'competition categories.')

        # Model args
        self.parser.add_argument('--config_path',
                                 dest='model_args.config_path',
                                 type=str, default=None)
        self.parser.add_argument('--calibrate',
                                 dest='model_args.calibrate',
                                 type=util.str_to_bool, default=False,
                                 help='Compute calibrated probabilities.')
        
        # TODO: Somehow need this line
        self.parser.add_argument('--moco', dest='model_args.moco',
                                 type=util.str_to_bool, default=True,
                                 help='Using moco')