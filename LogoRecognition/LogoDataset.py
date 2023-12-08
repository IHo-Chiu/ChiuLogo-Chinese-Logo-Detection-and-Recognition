import os
import torch
import random
import numpy as np
from PIL import Image 
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import get_paths
import lightning as L

class LogoDataset(Dataset):
    def __init__(self, dir, split='test', type='all', transform=None):
        self.dir = dir
        self.split = split # train / valid / test
        self.type = type # query / gallery / all
        self.transform = transform

        # get split classes
        self.classes = []
        for s in ['train', 'valid', 'test']:
            if s in self.split:
                split_file = os.path.join(self.dir, f'{s}.txt')
                with open(split_file, 'r') as f:
                    self.classes += f.read().split('\n')

        random.seed(5201314)
        random.shuffle(self.classes)
        
        # get all data in classes
        self.data = []
        for idx, label in enumerate(self.classes):
            # if idx > 20: break
            class_dir = os.path.join(self.dir, label)
            image_paths = get_paths(class_dir, ['jpg'])
            random.shuffle(image_paths)
            for i, image_path in enumerate(image_paths):
                if self.type == 'query':
                    if i >= 10: break
                elif self.type == 'gallery':
                    if i < 10: continue
                        
                self.data.append([idx, image_path])

        random.shuffle(self.data)

    def __str__(self):
        return f'({self.dir}/{self.split}/{self.type})' 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label, image_path = data
        
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        if self.split == 'test':
            return label, image, image_path
        else:
            return label, image

    def collate_fn(self, data):
        if self.split == 'test':
            labels = []
            images = []
            image_paths = []
            for label, image, image_path in data:
                labels.append(label)
                images.append(image)
                image_paths.append(image_path)
    
            return torch.tensor(labels), torch.stack(images), image_paths
        else:
            labels = []
            images = []
            for label, image in data:
                labels.append(label)
                images.append(image)
    
            return torch.tensor(labels), torch.stack(images)
        

class LogoDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir = "./", transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transform

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_set = LogoDataset(dir=self.data_dir, split='train', transform=self.transform)
            self.valid_set = LogoDataset(dir=self.data_dir, split='valid', transform=self.transform)
            print(f'train size: {len(self.train_set)}')
            print(f'valid size: {len(self.valid_set)}')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, 
                        collate_fn=self.train_set.collate_fn, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, 
                        collate_fn=self.valid_set.collate_fn, pin_memory=True, drop_last=False)
        

