import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')


INDEX = '32_2'

'''
tmux, index

'''

ACT_CLIP_P = 0.001

PRETRAINED = 'True'
MIXED = 'True'
STAGE = ''

METHOD = 'uniform'# uniform
OOD_METHOD = 'quant' # msp, odin, energy, react, vim
OOD_dataset = 'Places' 
# iNaturalist, SUN, Places, texture, 
# iSUN, LSUN, LSUN_resize, SVHN

# SEED = 12345 
# 1. training: main.py [cifar-10]
# 2. fine-tune: finetune.py [cifar-10]


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'Dataset to train')
parser.add_argument('--ood_dataset', type = str, default = OOD_dataset, help = 'OOD Dataset')

parser.add_argument('--data_dir', type = str, default = '/home/ta/research/ood/dataset/id_ood_datasets', help = 'The directory where the input data is stored.')
#parser.add_argument('--data_dir', type = str, default = '/media/ta/e9cf3417-0c3e-4e6a-b63c-4401fabeabc8/ta/id_ood_datasets', help = 'The directory where the input data is stored.')
#parser.add_argument('--data_dir', type = str, default = '/media/disk3/tachen/datasets/id_datasets', help = 'The directory where the input data is stored.')
#parser.add_argument('--data_dir', type = str, default = '/home/tachen/dataset/id_datasets', help = 'The directory where the input data is stored.')
parser.add_argument('--job_dir', type = str, default = f'experiment/{METHOD}/resnet/t_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--score_dir', type = str, default = f'score/{OOD_METHOD}/{OOD_dataset}/t_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--eval_dir', type = str, default = f'eval/{OOD_METHOD}/{OOD_dataset}/t_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'

parser.add_argument('--csv_dir', type = str, default = 'data/test_data_files/', help = 'The directory where the input data is stored.')
parser.add_argument('--csv_dir_mixed', type = str, default = 'data/test_mixed_data_files/', help = 'The directory where the input data is stored.')

parser.add_argument('--stage', type = str, default = STAGE, help = 'Load pruned model')

parser.add_argument('--pretrained', type = str, default = PRETRAINED, help = 'Load pruned model')
parser.add_argument('--mixed', type = str, default = MIXED, help = 'Load pruned model')

parser.add_argument('--method', type = str, default = METHOD, help = 'Load pruned model')
parser.add_argument('--ood_method', type = str, default = OOD_METHOD, help = 'Load pruned model')

parser.add_argument('--source_dir', type = str, default = 'pretrained/', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = 'densenet/checkpoint_100.pth.tar', help = 'The file the teacher model weights saved as.')
#parser.add_argument('--source_file', type = str, default = 'densenet/t_32_0/checkpoint/model_best.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None


## Training
parser.add_argument('--bitW', type = int, default = 32, help = 'Quantized bitwidth.') # None
parser.add_argument('--abitW', type = int, default = 16, help = 'Quantized bitwidth.') # None

parser.add_argument('--target_model', type = str, default = 'DenseNet3', help = 'The target model.')
parser.add_argument('--num_epochs', type = int, default = 15, help = 'The num of epochs to train.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--num_classes', type = int, default = 100, help = 'Number of classes.')
parser.add_argument('--train_batch_size', type = int, default = 64, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 64, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.004) 
parser.add_argument('--lr_G', type = float, default = 1e-3, help='learning rate (default: 0.001)')# 1. train: default = 0.01
# 2. fine_tuned: default = 5e-2
parser.add_argument('--lr_gamma', type = float, default = 0.1)
parser.add_argument('--ll_lambda', type = float, default = 100) 

parser.add_argument('--lr_decay_step', type = int, default = 30)
parser.add_argument('--lr_decay_steps', type = list, default = [10])

# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'The weight decay of loss.')

parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')

parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')

parser.add_argument('--act_clip_p', type = float, default = ACT_CLIP_P, help = 'Scale the sigmoid function.')


parser.add_argument('--p-w', default=None, type= int, help='weight sparsity level')
parser.add_argument('--p-a', default=10, type= int, help='activation sparsity level')

parser.add_argument('--clip_threshold', default=0.9, type=float, help='odin mahalanobis')

## Status
parser.add_argument('--print_freq', type = int, default = 500, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

