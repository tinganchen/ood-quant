import os
import numpy as np
import utils.common as utils
from utils.options import args
import utils.optimizer as opt
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from data import imagenet_data_train

import warnings

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")
#device = torch.device("cpu")


def main():

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0
    
    # loggers
    checkpoint = utils.checkpoint(args, '')
    print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
    writer_train = SummaryWriter(args.job_dir + '/run/train')
    writer_test = SummaryWriter(args.job_dir + '/run/test')

    # Data loading
    print('=> Preparing data..')
    loader = imagenet_data_train.Data(args)
    
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
        
       
    models = [model_t]

   
    param_t = [param for name, param in model_t.named_parameters()]
 
    optimizer_t = optim.SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
  
    scheduler_t = MultiStepLR(optimizer_t, args.lr_decay_steps, gamma = args.lr_gamma)
  

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_t.load_state_dict(ckpt['state_dict'])

        optimizer_t.load_state_dict(ckpt['optimizer'])

        scheduler_t.load_state_dict(ckpt['scheduler'])
        
        print('=> Continue from epoch {}...'.format(start_epoch))

    '''
    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return
    '''

    optimizers = [optimizer_t]
    schedulers = [scheduler_t]


    test_prec1 = 0.
    test_prec5 = 0.
    best_prec1 = 0.
    best_prec5 = 0.
    
    # Start training
    print('=> Start training..')
    
    
    num_epochs = args.num_epochs #if task_id > 0 else int(args.num_epochs//10)
    
    for epoch in range(start_epoch, num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, loader.loader_train, models, optimizers, epoch, writer_train, print_logger)
        
        test_prec1, test_prec5 = test(args, loader.loader_test, model_t, epoch, writer_test, print_logger)             
            
        is_best = best_prec1 < test_prec1
        
        if is_best:
            best_prec1 = test_prec1
            best_prec5 = test_prec5
    
        state = {
            'state_dict': model_t.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            
            'optimizer': optimizer_t.state_dict(),
            'scheduler': scheduler_t.state_dict(),
    
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
    
    print_logger.info(f"Best@prec1: {best_prec1:.2f} @prec5: {best_prec5:.2f}")


       
def train(args, loader_train, models, optimizers, epoch, writer_train, print_logger):
    losses_t = utils.AverageMeter()
 
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = models[0]
       
    cross_entropy = nn.CrossEntropyLoss()

    optimizer_t = optimizers[0]
    
    # switch to train mode
    model_t.train()
        
    num_iterations = len(loader_train)    
  
    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i
   
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        if i % 2 == 0:
            optimizer_t.zero_grad()
    
        ## train weights
        output_t = model_t(inputs).to(device)
            
        error_t = cross_entropy(output_t, targets)
        
        losses_t.update(error_t.item(), inputs.size(0))        
        
        error_t.backward() 
        
        writer_train.add_scalar('Performance_loss', error_t.item(), num_iters)
        
        if i % 2 == 0:
            optimizer_t.step()


        ## evaluate
        prec1, prec5 = utils.accuracy(output_t, targets, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
        writer_train.add_scalar('Train-top-1', top1.avg, num_iters)
        writer_train.add_scalar('Train-top-5', top5.avg, num_iters)
        
        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'       
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses_t,
                top1 = top1, 
                top5 = top5))
            
            
def test(args, loader_test, model_t, epoch, writer_test, print_logger):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_t.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_t(inputs.to(device))
            loss = cross_entropy(logits, targets)
            
            writer_test.add_scalar('Test_loss', loss.item(), num_iters)
        
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Test-top-1', top1.avg, num_iters)
            writer_test.add_scalar('Test-top-5', top5.avg, num_iters)
        
    print_logger.info(f'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '=======================================================\n'
                      .format(top1 = top1, top5 = top5))

    return top1.avg, top5.avg
    

if __name__ == '__main__':
    main()

