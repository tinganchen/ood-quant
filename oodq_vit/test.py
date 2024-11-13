import os

import utils.common as utils
from utils.options import args

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from data import data

import pandas as pd

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))


def main():

    # Data loading
    print('=> Loading preprocessed data..')
    
    loader = data.DataLoading(args)
    loader.load()

    loader_test = loader.loader_test
    
    # Create model
    
    print('=> Building model...')

    model = torch.load(args.pretrain_arch)
    
   
    if args.pretrained == 'True':
        # Load pretrained weights
        ckpt = torch.load(args.pretrain_dir + args.pretrain_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]

        new_state_dict = dict()
            
        for k, v in model.state_dict().items():
            new_state_dict[k] = state_dict[k]

        model.load_state_dict(new_state_dict)
        
        model = model.to(device)
   

   
    best_acc = inference(args, loader_test, model)

    print_logger.info(f"Best @test_acc: {best_acc:.3f}")

    
       
def train(args, loader_train, model, optimizer, epoch):
    losses = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    

    cross_entropy = nn.CrossEntropyLoss()
    
    # switch to train mode
    model.train()
 
    num_iterations = len(loader_train)
    
    for i, (inputs, targets, _) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
    
        ## train weights
        output = model(inputs).to(device)
        
        error = cross_entropy(output, targets)

        error.backward() # retain_graph = True
        
        losses.update(error.item(), inputs.size(0))
        
    
        ## step forward
        optimizer.step()
     

        ## evaluate
        prec1, prec5 = utils.accuracy(output, targets, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
 
        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train_acc {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses, 
                top1 = top1))
        
            
def inference(args, loader_test, model):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()
    
    
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, data_file) in enumerate(loader_test, 1):

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs).to(device)
            loss = cross_entropy(logits, targets)
            

            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
           
    print_logger.info('Test_acc {top1.avg:.3f}\n'
                 '====================================\n'
                 .format(top1 = top1))
    
    
    return top1.avg
        


if __name__ == '__main__':
    main()

