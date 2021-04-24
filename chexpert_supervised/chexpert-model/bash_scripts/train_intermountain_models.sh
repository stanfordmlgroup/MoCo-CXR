python train.py --dataset chexpert --batch_size 48 --iters_per_print 48 --iters_per_visual 48000 --iters_per_eval=4800 --iters_per_save=4800 --gpu_ids 0,1,2 --experiment_name DenseNet121_320_1e-04_uncertainty_ignore_top10 --num_epochs=3 --metric_name chexpert-competition-avg-AUROC --maximize_metric True --scale 320 --save_dir /deep/group/CheXpert/final_ckpts --max_ckpts 10 --keep_topk True