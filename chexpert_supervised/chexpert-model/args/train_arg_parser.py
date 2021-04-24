"""Define class for processing training command-line arguments."""
from .base_arg_parser import BaseArgParser
import util


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        # Model args
        self.parser.add_argument('--model',
                                 dest='model_args.model',
                                 choices=('DenseNet121', 'ResNet152',
                                          'Inceptionv4', 'ResNet18',
                                          'ResNet50',
                                          'ResNet34', 'ResNeXt101',
                                          'SEResNeXt101', 'NASNetA',
                                          'SENet154', 'MNASNet'),
                                 default='DenseNet121',
                                 help='Model name.')
        self.parser.add_argument('--pretrained', dest='model_args.pretrained',
                                 type=util.str_to_bool, default=True,
                                 help='Use a pretrained network.')

        self.parser.add_argument('--moco', dest='model_args.moco',
                                 type=util.str_to_bool, default=True,
                                 help='Using moco')

        self.parser.add_argument('--fine_tuning', dest='model_args.fine_tuning',
                                 type=str, default='None',
                                 help='Layer to fine tune')


        # Logger args
        self.parser.add_argument('--experiment_name',
                                 dest='logger_args.experiment_name',
                                 type=str, default='debugging',
                                 help='Experiment name.')
        self.parser.add_argument('--train_custom_csv',
                                 dest='data_args.csv',
                                 type=str, default=None,
                                 help='csv for custom dataset.')
        self.parser.add_argument('--val_custom_csv',
                                 dest='data_args.csv_dev',
                                 type=str, default=None,
                                 help='csv for custom dev dataset.')
        self.parser.add_argument('--iters_per_print',
                                 dest='logger_args.iters_per_print',
                                 type=int, default=64,
                                 help=('Number of iterations between ' +
                                       'printing loss to the console and ' +
                                       'TensorBoard.'))
        self.parser.add_argument('--iters_per_save',
                                 dest='logger_args.iters_per_save',
                                 type=int, default=8192,
                                 help=('Number of iterations between ' +
                                       'saving a checkpoint to save_dir.'))
        self.parser.add_argument('--iters_per_eval',
                                 dest='logger_args.iters_per_eval',
                                 type=int, default=8192,
                                 help=('Number of iterations between ' +
                                       'evaluations of the model.'))
        self.parser.add_argument('--iters_per_visual',
                                 dest='logger_args.iters_per_visual',
                                 type=int, default=16384,
                                 help=('Number of iterations between ' +
                                       'visualizing training examples.'))
        self.parser.add_argument('--max_ckpts',
                                 dest='logger_args.max_ckpts',
                                 type=int, default=10,
                                 help=('Number of checkpoints to keep ' +
                                       'before overwriting old ones.'))
        self.parser.add_argument('--keep_topk',
                                 dest='logger_args.keep_topk',
                                 type=util.str_to_bool, default=True,
                                 help=('Keep the top K checkpoints instead ' +
                                       'of most recent K checkpoints.'))

        # Training args
        self.parser.add_argument('--num_epochs',
                                 dest='optim_args.num_epochs',
                                 type=int, default=50,
                                 help=('Number of epochs to train. If 0, ' +
                                       'train forever.'))
        self.parser.add_argument('--metric_name',
                                 dest='optim_args.metric_name',
                                 choices=('chexpert-log_loss',
                                          'cxr14-log_loss',
                                          'chexpert-competition-log_loss',
                                          'chexpert-competition-AUROC',
                                          'shenzhen-AUROC',
                                          'chexpert-competition-single-AUROC',),
                                 default='chexpert-competition-AUROC',
                                 help=('Validation metric to optimize.'))
        self.parser.add_argument('--maximize_metric',
                                 dest='optim_args.maximize_metric',
                                 type=util.str_to_bool, default=True,
                                 help=('If True, maximize the metric ' +
                                       'specified by metric_name. ' +
                                       'Otherwise, minimize it.'))
        # Optimizer
        self.parser.add_argument('--optimizer',
                                 dest='optim_args.optimizer',
                                 type=str, default='adam',
                                 choices=('sgd', 'adam'), help='Optimizer.')
        self.parser.add_argument('--sgd_momentum',
                                 dest='optim_args.sgd_momentum',
                                 type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--sgd_dampening',
                                 dest='optim_args.sgd_dampening',
                                 type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--weight_decay',
                                 dest='optim_args.weight_decay',
                                 type=float, default=0.0,
                                 help='Weight decay (L2 coefficient).')
        # Learning rate
        self.parser.add_argument('--lr',
                                 dest='optim_args.lr',
                                 type=float, default=1e-4,
                                 help='Initial learning rate.')
        self.parser.add_argument('--lr_scheduler',
                                 dest='optim_args.lr_scheduler',
                                 type=str, default=None,
                                 choices=(None, 'step', 'multi_step',
                                          'plateau'),
                                 help='LR scheduler to use.')
        self.parser.add_argument('--lr_decay_gamma',
                                 dest='optim_args.lr_decay_gamma',
                                 type=float, default=0.1,
                                 help=('Multiply learning rate by this ' +
                                       'value every LR step (step and ' +
                                       'multi_step only).'))
        self.parser.add_argument('--lr_decay_step',
                                 dest='optim_args.lr_decay_step',
                                 type=int, default=100,
                                 help=('Number of epochs between each ' +
                                       'multiply-by-gamma step.'))
        self.parser.add_argument('--lr_milestones',
                                 dest='optim_args.lr_milestones',
                                 type=str, default='50,125,250',
                                 help=('Epochs to step the LR when using ' +
                                       'multi_step LR scheduler.'))
        self.parser.add_argument('--lr_patience',
                                 dest='optim_args.lr_patience',
                                 type=int, default=2,
                                 help=('Number of stagnant epochs before ' +
                                       'stepping LR.'))
        # Loss function
        self.parser.add_argument('--loss_fn',
                                 dest='optim_args.loss_fn',
                                 choices=('cross_entropy',),
                                 default='cross_entropy',
                                 help='loss function.')

        # Transform arguments
        self.parser.add_argument('--scale',
                                 dest='transform_args.scale',
                                 default=320, type=int)
        self.parser.add_argument('--crop',
                                 dest='transform_args.crop',
                                 type=int, default=320)
        self.parser.add_argument('--normalization',
                                 dest='transform_args.normalization',
                                 default='imagenet',
                                 choices=('imagenet', 'chexpert_norm'))
        self.parser.add_argument('--maintain_ratio',
                                 dest='transform_args.maintain_ratio',
                                 type=util.str_to_bool, default=True)

        # Data augmentation
        self.parser.add_argument('--rotate_min',
                                 dest='transform_args.rotate_min',
                                 type=float, default=0)
        self.parser.add_argument('--rotate_max',
                                 dest='transform_args.rotate_max',
                                 type=float, default=0)
        self.parser.add_argument('--rotate_prob',
                                 dest='transform_args.rotate_prob',
                                 type=float, default=0)
        self.parser.add_argument('--contrast_min',
                                 dest='transform_args.contrast_min',
                                 type=float, default=0)
        self.parser.add_argument('--contrast_max',
                                 dest='transform_args.contrast_max',
                                 type=float, default=0)
        self.parser.add_argument('--contrast_prob',
                                 dest='transform_args.contrast_prob',
                                 type=float, default=0)
        self.parser.add_argument('--brightness_min',
                                 dest='transform_args.brightness_min',
                                 type=float, default=0)
        self.parser.add_argument('--brightness_max',
                                 dest='transform_args.brightness_max',
                                 type=float, default=0)
        self.parser.add_argument('--brightness_prob',
                                 dest='transform_args.brightness_prob',
                                 type=float, default=0)
        self.parser.add_argument('--sharpness_min',
                                 dest='transform_args.sharpness_min',
                                 type=float, default=0)
        self.parser.add_argument('--sharpness_max',
                                 dest='transform_args.sharpness_max',
                                 type=float, default=0)
        self.parser.add_argument('--sharpness_prob',
                                 dest='transform_args.sharpness_prob',
                                 type=float, default=0)
        self.parser.add_argument('--horizontal_flip_prob',
                                 dest='transform_args.horizontal_flip_prob',
                                 type=float, default=0)
