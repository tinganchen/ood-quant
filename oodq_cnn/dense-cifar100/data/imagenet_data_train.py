from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

class ImageNet(Dataset):
    def __init__(self, args, root, train, transform):
        self.args = args
        self.root = root
        
        if train == 'train':
            data_csv = os.path.join(self.root, 'cifar100_train.csv')
        else:
            data_csv = os.path.join(self.root, 'cifar100_valid.csv')

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
        
        scale_size = 224
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(scale_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            normalize,
        ])
        
 
        trainset = ImageNet(args=args, root=args.csv_dir, train='train', 
                            transform=transform_train)
        
        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=0, pin_memory=pin_memory
            )
    

        
        testset = ImageNet(args=args, root=args.csv_dir, train='valid', 
                           transform=transform_test)
        
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=0, pin_memory=pin_memory
            )
        
  