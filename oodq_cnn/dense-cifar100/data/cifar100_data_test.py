import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

class OOD(Dataset):
    def __init__(self, args, root, transform):
        self.args = args
        self.root = root
        
        data_csv = os.path.join(self.root, f'{args.ood_dataset}_test.csv')
        
        self.data = pd.read_csv(data_csv)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data.iloc[i]['image'], self.data.iloc[i]['label']
        path = os.path.join(self.args.data_dir, path)
        image = self.transform(Image.open(path).convert('RGB'))
        return image, -1
        
        
class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False
  
        transform_test = transforms.Compose([
            #transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        '''
        transform_test_largescale = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform = transform_test if args.ood_dataset == 'texture' else transform_test_largescale
  
        id_testset = torchvision.datasets.CIFAR100(root='./data', 
                                                   train=False, download=True, 
                                                   transform=transform_test)
        
        self.loader_test_id = DataLoader(
            id_testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=0, pin_memory=pin_memory
            )
        
        ood_testset = OOD(args=args, root=args.csv_dir, 
                          transform=transform)
        
        self.loader_test_ood = DataLoader(
            ood_testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=0, pin_memory=pin_memory
            )
        
        mixed_testset = ConcatDataset([id_testset, ood_testset])
        
        self.loader_test_mixed = DataLoader(
            mixed_testset, batch_size=args.eval_batch_size, shuffle=True, 
            num_workers=0, pin_memory=pin_memory
            )
        
        
  