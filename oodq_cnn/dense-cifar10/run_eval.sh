# quant, iNaturalist, 32_16
python3 cal_val_score.py --gpus 0 --csv_dir data/train_data_files/ --method uniform  --ood_dataset iNaturalist --job_dir experiment/uniform/resnet/t_32_16/ --score_dir score/quant/iNaturalist/t_32_16/ --eval_dir eval/quant/iNaturalist/t_32_16/ --method uniform --ood_method quant --bitW 32 --abitW 16 --mixed False

