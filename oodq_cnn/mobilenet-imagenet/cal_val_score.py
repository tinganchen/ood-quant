import os
import numpy as np
import utils.common as utils
from utils.options import args
import utils.score2 as scr
from tensorboardX import SummaryWriter
from importlib import import_module
from sklearn.linear_model import LogisticRegressionCV

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from data import imagenet_data_val

import warnings

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")
#device = torch.device("cpu")


def main():
    
    args.job_dir = args.score_dir
    
    # loggers
    checkpoint = utils.checkpoint(args, '')
    print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))


    # Data loading
    print('=> Preparing data..')
    loader = imagenet_data_val.Data(args)
    
    # Create model
    print('=> Building model...')

    
    if args.method != 'our_q':
        ARCH = 'mobilenetv2'
    else: 
        ARCH = f'mobilenetv2_{args.method}'
    
    model_t = import_module(f'model.{ARCH}').__dict__[args.target_model]().to(device)
    
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
    
    #test_prec1, test_prec5 = test(args, loader.loader_test, model_t, print_logger) 
    
    #print_logger.info(f"Validation Best@prec1: {test_prec1:.2f} @prec5: {test_prec5:.2f}")

    
    # Start scoring
    print('=> Start scoring..')
    
    score(args, loader.loader_test, model_t)
    
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

def get_score(inputs, model, clip_l=None, clip=None, m=None, s=None, alpha=None, logits=None):
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
    elif args.ood_method == "quant":
        scores = scr.get_quant_score(inputs, model, clip_l, clip, m, s)
    #elif args.ood_method == "mahalanobis":
    #    scores = scr.get_mahalanobis_score(inputs, model)
    return scores

def score(args, loader_test, model_t):

    # switch to eval mode
    model_t.eval()

    #with torch.no_grad():
    for i, (inputs, _) in enumerate(loader_test, 1):
  
        inputs = inputs.to(device)
        
        if args.ood_method == 'react':
            scores, clip = get_score(inputs, model_t, clip=None, logits=None)
        elif args.ood_method == 'vim':
            scores, alpha = get_score(inputs, model_t, alpha=None, logits=None)
        elif args.ood_method == 'quant':
            scores, clip_l, clip, m, s = get_score(inputs, model_t, clip_l=None, clip=None, m=None, s=None, logits=None)
        else:
            scores = get_score(inputs, model_t)
        
        with open(os.path.join(args.job_dir, f"out_scores_{args.ood_method}.txt"), 'a') as f:
            for score in scores:
                f.write("{}\n".format(score))
        
        if args.ood_method == 'react':
            with open(os.path.join(args.job_dir, f"out_clip_{args.ood_method}.txt"), 'a') as f:
                f.write("{}\n".format(clip))
        elif args.ood_method == 'vim':
            with open(os.path.join(args.job_dir, f"out_alpha_{args.ood_method}.txt"), 'a') as f:
                f.write("{}\n".format(alpha))
        elif args.ood_method == 'quant':

            with open(os.path.join(args.job_dir, f"out_quant_clip_l_{args.ood_method}.txt"), 'a') as f:
                f.write("{}\n".format(clip_l))
                
            with open(os.path.join(args.job_dir, f"out_quant_clip_{args.ood_method}.txt"), 'a') as f:
                f.write("{}\n".format(clip))
            with open(os.path.join(args.job_dir, f"out_m_{args.ood_method}.txt"), 'a') as f:
                f.write("{}\n".format(m))
            with open(os.path.join(args.job_dir, f"out_s_{args.ood_method}.txt"), 'a') as f:
                f.write("{}\n".format(s))

    return 0

if __name__ == '__main__':
    
    main()

