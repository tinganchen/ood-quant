# quant, iSUN, 16_lam0
python3 cal_val_score.py --gpus 0 --csv_dir data/train_data_files/ --method uniform  --ood_dataset iSUN --job_dir experiment/uniform/resnet/t_16_lam0/ --score_dir score/quant/iSUN/t_16_lam0/ --eval_dir eval/quant/iSUN/t_16_lam0/ --method uniform --ood_method quant --bitW 32 --abitW 16 --mixed True --lam 0.


