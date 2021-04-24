
USER='minhphu'
ROOT=/deep/group/${USER}
TEMP=${ROOT}/dump

cp /deep/group/CheXpert/final_ckpts/CheXpert-Ignore/best.pth.tar $TEMP
cp /deep/group/CheXpert/final_ckpts/CheXpert-Ignore/args.json $TEMP
cd ${ROOT}/aihc-winter19-robustness/chexpert-model/
python test.py --inference_only True \
               --dataset custom \
               --together True \
               --test_csv /deep/group/chexperturbed/data/toy_of_CheXpert/train.csv \
               --ckpt_path ${TEMP}/best.pth.tar \
               --phase test \
               --save_dir $TEMP \

               
