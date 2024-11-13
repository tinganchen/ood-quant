import argparse
import os 

parser = argparse.ArgumentParser(description = 'Vision Transformer')

QUANTIZED = 'True'
INDEX = '0'
METHOD = 'uniform' # data_free_kd, kd_kl


# SEED = 12345 
# 1. training: main.py [cifar-10]
# 2. fine-tune: finetune.py [cifar-10]


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')

parser.add_argument('--pretrain_data_dir', type = str, default = os.getcwd() + '/data/', help = 'The directory where the input data is stored.')

parser.add_argument('--train_dataset', type = str, default = None, help = 'Dataset to train')
parser.add_argument('--test_dataset', type = str, default = os.getcwd() + '/../../hw3_data/'+'p1_data/val', help = 'Dataset to validate or test')
parser.add_argument('--test_only', type = str, default = 'True', help = 'Test only?') 

parser.add_argument('--model', type = str, default = 'ViT', help = 'The model.')
parser.add_argument('--method', type = str, default = METHOD, help = 'Load quantized model')

parser.add_argument('--job_dir', type = str, default = f'experiment/vit/output_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--output_file', type = str, default = 'result.csv', help = 'The directory where the summaries will be stored.') # 'experiments/'


parser.add_argument('--quantized', type = str, default = QUANTIZED, help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--pretrain_dir', type = str, default = '', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--pretrain_file', type = str, default = 'best_model/model_best.pt', help = 'The directory where the summaries will be stored.') # 'experiments/'
parser.add_argument('--pretrain_arch', type = str, default = 'best_model/model.pt', help = 'The directory where the summaries will be stored.') # 'experiments/'
# 1. train: default = f'experiment/resnet/t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'
# 2. fine_tuned: default = f'experiment/resnet/ft_thres_{THRES}_t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/mobile/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None


## Training
parser.add_argument('--bitW', type = int, default = 4, help = 'Quantized bitwidth.') # None
parser.add_argument('--abitW', type = int, default = 4, help = 'Quantized bitwidth.') # None

parser.add_argument('--num_classes', type = int, default = 37, help = 'The num of classes to train.') # 100
parser.add_argument('--num_epochs', type = int, default = 50, help = 'The num of epochs to train.') # 100

parser.add_argument('--img_size', type = int, default = 384, help = 'The num of classes to train.') # 100
parser.add_argument('--patch_size', type = int, default = 32, help = 'The num of classes to train.') # 100
parser.add_argument('--dim', type = int, default = 64, help = 'The num of classes to train.') # 100
parser.add_argument('--mlp_dim', type = int, default = 128, help = 'The num of classes to train.') # 100
parser.add_argument('--depth', type = int, default = 16, help = 'The num of classes to train.') # 100
parser.add_argument('--heads', type = int, default = 12, help = 'The num of classes to train.') # 100

# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 32, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 32, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.0005)
# 1. train: default = 0.1
# 2. fine_tuned: default = 5e-2

parser.add_argument('--lr_decay_step',type = int, default = 20)
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--weight_decay', type = float, default = 2e-4, help = 'The weight decay of loss.')

## Status
parser.add_argument('--print_freq', type = int, default = 20, help = 'The frequency to print loss.')


args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))
