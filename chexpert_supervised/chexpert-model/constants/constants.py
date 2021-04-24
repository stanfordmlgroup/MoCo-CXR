"""Define constants to be used throughout the repository."""
from pathlib import Path

# Main directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = Path("/deep/group")

# Datasets
CHEXPERT = "chexpert"
CUSTOM = "custom"
CHEXPERT_SINGLE = "chexpert_single_special"
CXR14 = "cxr14"
SHENZHEN = "shenzhen_special"

# Predict config constants
CFG_TASK2MODELS = "task2models"
CFG_AGG_METHOD = "aggregation_method"
CFG_CKPT_PATH = "ckpt_path"
CFG_IS_3CLASS = "is_3class"

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COL_PATH = "Path"
COL_STUDY = "Study"
COL_TASK = "Tasks"
COL_METRIC = "Metrics"
COL_VALUE = "Values"
TASKS = "tasks"
UNCERTAIN = -1
MISSING = -2

# CheXpert specific constants
CHEXPERT_DATASET_NAME = "CheXpert-v1.0"
CHEXPERT_PARENT_DATA_DIR = DATA_DIR / "CheXpert"
CHEXPERT_SAVE_DIR = CHEXPERT_PARENT_DATA_DIR / "models/"
CHEXPERT_DATA_DIR = CHEXPERT_PARENT_DATA_DIR / CHEXPERT_DATASET_NAME
CHEXPERT_TEST_DIR = CHEXPERT_PARENT_DATA_DIR / "CodaLab"
CHEXPERT_UNCERTAIN_DIR = CHEXPERT_PARENT_DATA_DIR / "Uncertainty"
CHEXPERT_RAD_PATH = CHEXPERT_PARENT_DATA_DIR / "rad_perf_test.csv"
CHEXPERT_MEAN = [.5020, .5020, .5020]
CHEXPERT_STD = [.085585, .085585, .085585]
CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"
                  ]
CHEXPERT_SINGLE_TASKS = ["No Finding",
                         "Pleural Effusion",
                        ]

CHEXPERT_COMPETITION_TASKS = ["Atelectasis",
                              "Cardiomegaly",
                              "Consolidation",
                              "Edema",
                              "Pleural Effusion"
                              ]
CHEXPERT_COMPETITION_SINGLE_TASKS = CHEXPERT_COMPETITION_TASKS
# CHEXPERT_COMPETITION_SINGLE_TASKS = ["Pleural Effusion"]

SHENZHEN_TASKS = ['Tuberculosis']

# CXR14 specific constants
CXR14_DATA_DIR = DATA_DIR / CXR14
CXR14_TASKS = ["Cardiomegaly",
               "Emphysema",
               "Pleural Effusion",
               "Hernia",
               "Infiltration",
               "Mass",
               "Nodule",
               "Atelectasis",
               "Pneumothorax",
               "Pleural Thickening",
               "Pneumonia",
               "Fibrosis",
               "Edema",
               "Consolidation"]
CALIBRATION_FILE = "calibration_params.json"

DATASET2TASKS = {CHEXPERT: CHEXPERT_TASKS,
                 CUSTOM: CHEXPERT_TASKS,
                 CHEXPERT_SINGLE: CHEXPERT_TASKS,
                 CXR14: CXR14_TASKS,
                 SHENZHEN: SHENZHEN_TASKS}

EVAL_METRIC2TASKS = {'chexpert-log_loss': CHEXPERT_TASKS,
                     'cxr14-log_loss': CXR14_TASKS,
                     'shenzhen-AUROC': SHENZHEN_TASKS,
                     'chexpert-competition-log_loss': CHEXPERT_COMPETITION_TASKS,
                     'chexpert-competition-AUROC': CHEXPERT_COMPETITION_TASKS,
                     'chexpert-competition-single-AUROC': CHEXPERT_COMPETITION_TASKS}

NamedTasks = {'chexpert': CHEXPERT_TASKS,
        'chexpert-competition': CHEXPERT_COMPETITION_TASKS,
        'pleural-effusion': CHEXPERT_TASKS
        }
