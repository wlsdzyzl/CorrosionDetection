import torch, numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset

import os
import glob
from itertools import chain

from PIL import Image
import random

def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

def test_collate(batch):
    return batch

class Rust(Dataset):
    def __init__(self, rust_paths, norust_paths, phase = 'train'):
        self.rust_paths = rust_paths
        self.norust_paths = norust_paths
        self.phase = phase
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.size = len(self.rust_paths) + len(self.norust_paths)
        self.images = []
        self.labels = None    
        for p in self.rust_paths:
            print(p)
            self.images.append( self.preprocess(Image.open(p)) )
        for p in self.norust_paths:
            print(p)
            self.images.append( self.preprocess(Image.open(p)) )
        if self.phase != 'test':
            self.labels = list([torch.LongTensor([1]) for _ in range(len(self.rust_paths))]) + list([torch.LongTensor([0]) for _ in range(len(self.norust_paths))])

        
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        if self.phase == 'test':
            return self.images[idx]
        return self.images[idx], self.labels[idx]

class RustAnnotation(Dataset):
    def __init__(self, file_path, raw_sub_path = 'raw', mask_sub_path = 'mask', phase = 'train'):
        self.file_path = file_path
        raw_paths = traverse_paths(self.file_path + '/' + raw_sub_path)
        mask_paths = traverse_paths(self.file_path + '/' + mask_sub_path)
        self.phase = phase
        self.preprocess = transforms.Compose([
            transforms.Resize(480),
            transforms.ToTensor()
        ])
        self.size = len(raw_paths)

        self.images = []
        self.labels = None
        for p in raw_paths:
            print(p)
            self.images.append( self.preprocess(Image.open(p)) / 255.0 )
        if self.phase != 'test':
            self.labels = []
            for p in mask_paths:
                self.labels.append( (self.preprocess(Image.open(p)) > 0).long() )

    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        if self.phase == 'test':
            return self.images[idx]
        return self.images[idx], self.labels[idx]
    
def traverse_paths(target_dir):
    results = []
    if os.path.isdir(target_dir):
        # if file path is a directory take all NII files in that directory
        iters = [sorted(glob.glob(os.path.join(target_dir, '*'))) ]
        for fp in chain(*iters):
            results.append(fp)
    else:
        print('Input path should be a directory.')
    return results

    
def get_rust_loader(loader_config, is_test = False):
    num_workers = loader_config.get('num_workers', 1)
    rust_sub_path = loader_config.get('rust_dir', 'rust')
    norust_sub_path = loader_config.get('norust_dir', 'norust')
    rust_paths = traverse_paths(loader_config['file_path'] + '/' + rust_sub_path) 
    norust_paths = traverse_paths(loader_config['file_path'] + '/' + norust_sub_path)
    random.shuffle(rust_paths)
    random.shuffle(norust_paths)
    train_ratio = loader_config.get('train_ratio', 0.75)

    train_rust_size = int(train_ratio * len(rust_paths))
    train_norust_size = int(train_ratio * len(norust_paths))

    batch_size = loader_config.get('batch_size', 1)

    if is_test:
        test_dataset = Rust(rust_paths = rust_paths[train_rust_size:], norust_paths = norust_paths[train_norust_size:], phase = 'test')
        return DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, collate_fn = test_collate)
    train_dataset = Rust(rust_paths = rust_paths[:train_rust_size], norust_paths = norust_paths[:train_norust_size], phase = 'train')
    val_dataset = Rust(rust_paths = rust_paths[train_rust_size:], norust_paths = norust_paths[train_norust_size:], phase = 'val')

    return DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn = custom_collate), \
        len(train_dataset), \
        DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, collate_fn = custom_collate), \
        len(val_dataset)

def get_annotated_rust_loader(loader_config):
    num_workers = loader_config.get('num_workers', 1)
    raw_sub_path = loader_config.get('raw_dir', 'raw')
    mask_sub_path = loader_config.get('mask_dir', 'mask')
    phase = loader_config.get('phase', 'train')
    dataset = RustAnnotation(file_path = loader_config['file_path'], raw_sub_path = raw_sub_path, mask_sub_path = mask_sub_path, phase = phase)
    batch_size = loader_config.get('batch_size', 1)
    if phase == 'test':
        return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, collate_fn = test_collate)
    return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = (phase=='train'), num_workers = num_workers, collate_fn = custom_collate), \
        len(dataset)