# quant, iNaturalist, 16_lam0.2
python3 cal_val_score.py --gpus 0 --csv_dir data/train_data_files/ --method uniform  --ood_dataset iNaturalist --job_dir experiment/uniform/resnet/t_16_lam0.2/ --score_dir ablation_score/quant/iNaturalist/t_16_lam0.2/ --eval_dir ablation_eval/quant/iNaturalist/t_16_lam0.2/ --method uniform --ood_method quant --bitW 32 --abitW 16 --mixed True --lam 0.2

