# aihc-winter19-robustness
Repo for project on robustness to medical images.
Development branch for conaug-2020.

## Activate environment
Default experiment to activate is defined in chexpert-model/sbatch/conaug/sbatch_commands/envs.py in `CONAUG_ENV` (line 23)

Point it to your own virtual environment if needed.
e.g.
```
source /deep/u/canliu/envs/aihc_chexpert/bin/activate
```

## chexpert-model
This directory is a fork of the original chexpert-model, in our organization.

### Usage
#### Automation.
##### See relevant code in chexpert-model/sbatch/conaug.

##### Generate sbatch scripts en masse:
1. Set up finetuning config by modifying *chexpert-model/sbatch/conaug/configs/finetune.json*.
2. Specify experiments to finetune with by entering experiment names in *CKPT_LIST* in *chexpert-model/sbatch/conaug/script_generation.py*.
3. Generate scripts: 
```
python chexpert-model/sbatch/conaug/script_generation.py
``` 
with optional arguments:
```
--user_id: owner of the pretrained checkpoints. Default: account running the script generation code.
--epochs: number of epochs. Default: calculated automatically based on label fractions.
--cpu: cpu per tasks. Default: 4.
--mem: cpu memory to request. Default: 32000.
--log_path: directory to log job status. Default: /sailhome/<user>/experiments.
```
Note that each group of finetuning experiments is associated with a unique timestamp.

##### Launch sbatch jobs:
1. Specify jobs to run by in *CONFIG* dictionary in *chexpert-model/sbatch/conaug/job_management.py*.
2. Launch jobs:
```
python chexpert-model/sbatch/conaug/job_management.py
```
with optional arguments:
```
--refresh: frequency to refresh screen (to print out current job status).
```

---------------
Following parts are informative but also could be outdated.

#### Training
Single model (default train and val set):
```
python train.py --dataset chexpert --save_dir <path to save dir> --experiment_name <type of model/experiment>
```
Ensemble model: Training the ensemble model consists of individually training 15 models separately. It may be good to use sbatch to train these models separately. 

Single model (custom train and val set): please use full paths to the images in the csvs if custom_dataset=True
```
python train.py --dataset custom  --train_custom_csv <path to train csv>  --val_custom_csv <path to val csv> --save_dir <path to save dir> --experiment_name <type of model/experiment>
```
(please look at train_arg_parser for other flags such as gpu, number of epochs, etc.)
  

#### Testing
Single model (default test set):
```
python test.py --dataset chexpert --ckpt_path <path to checkpoint> --phase {valid, test} --save_dir <path to save dir>
```
Ensemble (default test set):
```
python test.py --dataset chexpert --config_path <path to config> --phase {valid, test} --save_dir <path to save dir>
```
Single model (custom test set, separate test gt/paths): please use full paths to the images in the csvs if custom_dataset=True
```
python test.py --dataset custom --ckpt_path <path to checkpoint> --phase {valid, test} --save_dir <path to save dir> --test_groundtruth <path to gt labels of studies in test set> --test_image_paths <path to images paths in test set>
```
Single model (custom test set, test csv): please use full paths to the images in the csvs if custom_dataset=True
```
python test.py --dataset custom --ckpt_path <path to checkpoint> --phase {valid, test} --save_dir <path to save dir> --together True --test_csv <path to test csv, same format as train/val> 
```

(please look at test_arg_parser for other flags such as save_cams)

A note on CAMS generation:
```--save_cams True```: to generate CAMS
```--only_competition_cams True```: to only generate CAMS for competition classes
CAMS will only be generated for classes where groundtruth is 1.

### Reproduce CheXpert test results
`python test.py --dataset chexpert --config_path predict/config/final.json --phase test --save_dir <path to save dir>`

### Evaluating a pre-trained model
Some pre-trained models are available in `/deep/group/CheXpert/final_ckpts/`. We can try a 3-class model. Make a temporary folder `[temp]`, and do:
```
cp /deep/group/CheXpert/final_ckpts/CheXpert-3-class/best.pth.tar [temp]
cp /deep/group/CheXpert/final_ckpts/CheXpert-3-class/args.json [temp]
cd [repo]/chexpert-model/
python test.py --dataset chexpert --ckpt_path [temp]/best.pth.tar --phase {valid, test} --model_uncertainty True --save_dir <path to save dir>
```
Regarding the structure of the `[temp]` folder, let `[phase]` be the phase selected previously. Then, `[temp]/results/[phase]/scores.txt` contains a variety of metrics tabulated by the `Evaluator`. On the branch `mark_model_analysis`, `test.py` also saves `groundtruth.csv` and `predictions.csv` to `[temp]/results/[phase]/`.
