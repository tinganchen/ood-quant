import os
import numpy as np
import pandas as pd
import utils.common as utils
from utils.options import args
import utils.score as scr
from tensorboardX import SummaryWriter
from importlib import import_module
from sklearn.linear_model import LogisticRegressionCV

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")
#device = torch.device("cpu")


def main():
    
    args.job_dir = args.eval_dir
    
    # loggers
    checkpoint = utils.checkpoint(args, '')
    print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))

    
    # Calculate detection error
    print('=> Calculate OOD detection error..')
    
    ## get threshold from validation id dataset
    id_val_score_file = os.path.join(args.score_dir, f"out_scores_{args.ood_method}.txt")
    
    with open(id_val_score_file, 'r') as f:
        id_val_scores = f.readlines()
        
    id_val_scores = [float(s[:-1]) for s in id_val_scores]
    
    threshold = np.percentile(id_val_scores, 5) # 0.95 tpr
    
    if args.mixed == 'False':
        ## get id/ood scores from test datasets
        id_score_file = os.path.join(args.job_dir, f"out_scores_{args.dataset}-1k_{args.ood_method}.txt")
        
        ood_score_file = os.path.join(args.job_dir, f"out_scores_{args.ood_dataset}_{args.ood_method}.txt")
        
        with open(id_score_file, 'r') as f:
            id_test_scores = f.readlines()
            
        id_test_scores = [float(s[:-1]) for s in id_test_scores]
        
        with open(ood_score_file, 'r') as f:
            ood_test_scores = f.readlines()
            
        ood_test_scores = [float(s[:-1]) for s in ood_test_scores]
        
        fpr95, thres = utils.cal_fpr(id_test_scores, ood_test_scores, threshold)
        auroc, _, _ = utils.cal_auc(id_test_scores, ood_test_scores)
    
    else: #if args.ood_dataset == 'texture':
        ## get id/ood scores from test datasets
        mixed_test_score_file = os.path.join(args.job_dir, f"out_mixed_scores_{args.ood_dataset}_{args.ood_method}.txt")
        
        with open(mixed_test_score_file, 'r') as f:
            test_scores = f.readlines()
            
        test_scores = np.array([float(s[:-1]) for s in test_scores])
        
        data_csv = os.path.join(args.csv_dir_mixed, f'imagenet-1k_{args.ood_dataset}_test.csv')
        
        labels = pd.read_csv(data_csv)['label']
        
        id_test_scores = test_scores[labels[:len(test_scores)] == 1]
        ood_test_scores = test_scores[labels[:len(test_scores)] == 0]
        
        fpr95, thres = utils.cal_fpr(id_test_scores, ood_test_scores, threshold)
        auroc, _, _ = utils.cal_auc(id_test_scores, ood_test_scores)
    '''
    else:
        ## get id/ood scores from test datasets
        mixed_test_score_file = os.path.join(args.job_dir, f"out_mixed_scores_{args.ood_dataset}_{args.ood_method}.txt")
        
        with open(mixed_test_score_file, 'r') as f:
            test_scores = f.readlines()
            
        test_scores = np.array([float(s[:-1]) for s in test_scores])
        
        data_csv = os.path.join(args.csv_dir_mixed, f'imagenet-1k_{args.ood_dataset}_test.csv')
        
        labels = pd.read_csv(data_csv)['label']
        images = pd.read_csv(data_csv)['image']
        
        id_test_scores = test_scores[labels[:len(test_scores)] == 1]
        ood_test_scores = test_scores[labels[:len(test_scores)] == 0]
        
        ood_data = images[labels[:len(test_scores)] == 0]
        
        categories = np.unique(np.array([d.split('/')[3] for d in ood_data.tolist()]))
        
        fpr95 = 0.
        auroc = 0.
        for c in categories:
            ood_score = ood_test_scores[ood_data.str.contains(c)]
            fpr, thres = utils.cal_fpr(id_test_scores, ood_score, threshold)
            au, _, _ = utils.cal_auc(id_test_scores, ood_score)
        
            fpr95 += fpr  
            auroc += au
        
        fpr95 /= len(categories)
        auroc /= len(categories)
        '''
    print_logger.info(f"Testing Best@fpr95: {fpr95:.4f} @auroc: {auroc:.4f}")
    #print(f"Testing Best@fpr95: {fpr95:.4f} @auroc: {auroc:.4f}")

    
if __name__ == '__main__':
    
    main()

