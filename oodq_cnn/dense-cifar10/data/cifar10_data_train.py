from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import torchvision

class ImageNet(Dataset):
    def __init__(self, args, root, train, transform):
        self.args = args
        self.root = root
        
        if train == 'train':
            data_csv = os.path.join(self.root, 'cifar10_train.csv')
        else:
            data_csv = os.path.join(self.root, 'cifar10_valid.csv')

        self.data = pd.read_csv(data_csv)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data.iloc[i]['image'], self.data.iloc[i]['label']
        path = os.path.join(self.args.data_dir, path)
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
        
        
class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False
        
        transform_test_largescale = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True, download=True, 
                                                transform=transform_test_largescale)
        
        
        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=0, pin_memory=pin_memory
            )
    

        
        testset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=False, download=True, 
                                                transform=transform_test_largescale)
        
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=0, pin_memory=pin_memory
            )
        
  