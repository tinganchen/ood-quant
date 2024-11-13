import os
import numpy as np
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

from data import imagenet_data_test

import warnings

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")
#device = torch.device("cpu")


def main():
    
    args.job_dir = args.eval_dir
    
    # loggers
    checkpoint = utils.checkpoint(args, '')
    print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))


    # Data loading
    print('=> Preparing data..')
    loader = imagenet_data_test.Data(args)
    
    # Create model
    print('=> Building model...')

    
    if args.method == 'our_q':
        ARCH = 'resnet'
    else: 
        ARCH = f'resnet_{args.method}'
    
    
    model_t = import_module(f'model.{ARCH}').__dict__[args.target_model](bitW = args.bitW, 
                                                                         abitW = args.abitW, 
                                                                         stage = args.stage).to(device)
        
    # Load pretrained weights
    if args.pretrained == 'True':
        pretrained_file = args.source_dir + args.source_file
       
        ckpt = torch.load(pretrained_file, map_location = device)
        state_dict = ckpt['state_dict']
    
        model_dict_t = model_t.state_dict()
        
        for name, param in state_dict.items():
            if name in list(model_dict_t.keys()):
                model_dict_t[name] = param
        
        model_t.load_state_dict(model_dict_t)
        model_t = model_t.to(device)
        
        del ckpt, state_dict, model_dict_t
        
    # Calculate classification accuracy first
    print('=> Calculate accuracy..')
    
    #test_prec1, test_prec5 = test(args, loader.loader_test_id, model_t, print_logger) 
    
    #print_logger.info(f"Testing Best@prec1: {test_prec1:.2f} @prec5: {test_prec5:.2f}")

    # Start scoring
    print('=> Start scoring..')
    
    if args.mixed == 'False':
        score(args, loader.loader_test_id, loader.loader_test_ood, model_t)
    else:
        score(args, None, loader.loader_test_mixed, model_t)
    
    print('Finish.')
    
def test(args, loader_test, model_t, print_logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to eval mode
    model_t.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
  
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_t(inputs.to(device))
  
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
  
    #print_logger.info(f'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
    #                  '=======================================================\n'
    #                  .format(top1 = top1, top5 = top5))

    return top1.avg, top5.avg     

def get_score(inputs, model, clip_l=None, clip=None, m=None, s=None, alpha=None, logits=None,
              thres=None, ood_m=None, ood_s=None, ood_n=None, id_m=None, id_s=None):
    if args.ood_method == "msp":
        scores = scr.get_msp_score(inputs, model)
    elif args.ood_method == "odin":
        scores = scr.get_odin_score(inputs, model)
    elif args.ood_method == "energy":
        scores = scr.get_energy_score(inputs, model)
    elif args.ood_method == "react":
        scores = scr.get_react_score(inputs, model, clip)
    elif args.ood_method == "vim":
        scores = scr.get_vim_score(inputs, model, alpha)
    elif args.ood_method == "quant" and args.mixed == 'False':
        scores = scr.get_quant_score(inputs, model, clip_l, clip, m, s)
    elif args.ood_method == "quant" and args.mixed == 'True':
        scores = scr.get_quant_mixed_score(inputs, model, clip_l, clip, m, s, 
                                           ood_m, ood_s, ood_n, id_m, id_s, thres)
    #elif args.ood_method == "mahalanobis":
    #    scores = scr.get_mahalanobis_score(inputs, model)
    return scores

def score(args, loader_test_id, loader_test_ood, model_t):
    
    # switch to eval mode
    model_t.eval()
    
    # load validation parameters
    if args.ood_method == 'react':
        with open(os.path.join(args.score_dir, f"out_clip_{args.ood_method}.txt"), 'r') as f:
            id_val_clip = f.readlines()

        id_val_clip = [float(s[:-1]) for s in id_val_clip]

    elif args.ood_method == 'vim':
        with open(os.path.join(args.score_dir, f"out_alpha_{args.ood_method}.txt"), 'r') as f:
            id_val_alpha = f.readlines()

        id_val_alpha = [float(s[:-1]) for s in id_val_alpha]

    elif args.ood_method == 'quant':
        with open(os.path.join(args.score_dir, f"out_quant_clip_l_{args.ood_method}.txt"), 'r') as f:
            id_val_clip_l = f.readlines()

        id_val_clip_l = [float(s[:-1]) for s in id_val_clip_l]
        
        with open(os.path.join(args.score_dir, f"out_quant_clip_{args.ood_method}.txt"), 'r') as f:
            id_val_clip = f.readlines()

        id_val_clip = [float(s[:-1]) for s in id_val_clip]
        
        with open(os.path.join(args.score_dir, f"out_m_{args.ood_method}.txt"), 'r') as f:
            id_val_m = f.readlines()

        id_val_m = [float(s[:-1]) for s in id_val_m]
        
        with open(os.path.join(args.score_dir, f"out_s_{args.ood_method}.txt"), 'r') as f:
            id_val_s = f.readlines()

        id_val_s = [float(s[:-1]) for s in id_val_s]
        
        
    # load validation threshold 
    id_val_score_file = os.path.join(args.score_dir, f"out_scores_{args.ood_method}.txt")
    
    with open(id_val_score_file, 'r') as f:
        id_val_scores = f.readlines()
        
    id_val_scores = [float(s[:-1]) for s in id_val_scores]
    
    val_score_threshold = np.percentile(id_val_scores, 5) # 0.95 tpr
    
    # scoring
    if args.mixed == 'False':
        # ID
        out_f1 = os.path.join(args.job_dir, f"out_scores_{args.dataset}-1k_{args.ood_method}.txt")
        
        if not os.path.isfile(out_f1):
            for _, (inputs, _) in enumerate(loader_test_id, 1):
          
                inputs = inputs.to(device)
                
                if args.ood_method == 'react':
                    s, _ = get_score(inputs, model_t, clip=np.mean(id_val_clip), logits=None)
                    
                elif args.ood_method == 'vim':
                    s, _ = get_score(inputs, model_t, alpha=np.mean(id_val_alpha), logits=None)
                    
                elif args.ood_method == 'quant':
                    s, _, _, _, _ = get_score(inputs, model_t, 
                                              clip_l=np.mean(id_val_clip_l), 
                                              clip=np.mean(id_val_clip), 
                                              m=np.mean(id_val_m),
                                              s=np.mean(id_val_s),
                                              logits=None)
                    
                else:
                    s = get_score(inputs, model_t, logits=None)
                
                with open(out_f1, 'a') as f:
                    for i in s:
                        f.write("{}\n".format(i))
        
        # OOD
        out_f2 = os.path.join(args.job_dir, f"out_scores_{args.ood_dataset}_{args.ood_method}.txt")
        
        if not os.path.isfile(out_f2):
            for _, (inputs, _) in enumerate(loader_test_ood, 1):
          
                inputs = inputs.to(device)
                
                if args.ood_method == 'react':
                    s, _ = get_score(inputs, model_t, clip=np.mean(id_val_clip), logits=None)
                    
                elif args.ood_method == 'vim':
                    s, _ = get_score(inputs, model_t, alpha=np.mean(id_val_alpha), logits=None)
                    
                elif args.ood_method == 'quant':
                    s, _, _, _, _ = get_score(inputs, model_t,
                                              clip_l=np.mean(id_val_clip_l),
                                              clip=np.mean(id_val_clip), 
                                              m=np.mean(id_val_m),
                                              s=np.mean(id_val_s),
                                              logits=None)
                else:
                    s = get_score(inputs, model_t, logits=None)
                
                with open(out_f2, 'a') as f:
                    for i in s:
                        f.write("{}\n".format(i))
    else:
        out_f = os.path.join(args.job_dir, f"out_mixed_scores_{args.ood_dataset}_{args.ood_method}.txt")
        
        out_f_ood_info = os.path.join(args.job_dir, f"out_ood_stats_{args.ood_dataset}_{args.ood_method}.txt")
        
        out_f_id_info = os.path.join(args.job_dir, f"out_id_stats_{args.ood_dataset}_{args.ood_method}.txt")
        
        
        ood_m, ood_s, ood_n, id_m, id_s = None, None, None, None, None
        
        if not os.path.isfile(out_f):
            for i, (inputs, _) in enumerate(loader_test_ood, 1):
          
                inputs = inputs.to(device)
                
                # args.ood_method == 'quant'
                result = get_score(inputs, model_t, 
                                   clip_l=np.mean(id_val_clip_l), 
                                   clip=np.mean(id_val_clip), 
                                   m=np.mean(id_val_m),
                                   s=np.mean(id_val_s),
                                   ood_m=ood_m, 
                                   ood_s=ood_s,
                                   ood_n=ood_n,
                                   id_m=id_m,
                                   id_s=id_s,
                                   thres=val_score_threshold)
               
                s, ood_m, ood_s, ood_n, id_m, id_s = result
                
                with open(out_f, 'a') as f:
                    for out_s in s:
                        f.write("{}\n".format(out_s))
                
                with open(out_f_ood_info, 'w') as f:
                    f.write("{}\n".format(ood_m))
                    f.write("{}\n".format(ood_s))
                
                with open(out_f_id_info, 'w') as f:
                    f.write("{}\n".format(id_m))
                    f.write("{}\n".format(id_s))

        
    return 0

    

if __name__ == '__main__':
    
    main()

