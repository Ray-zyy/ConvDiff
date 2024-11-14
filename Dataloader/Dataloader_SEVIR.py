import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as data_utils


class SevirDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = np.load(data_path)
        print(self.data.shape)
        self.transform = transform
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames = torch.tensor(self.data[idx][:12], dtype=torch.float32)
        output_frames = torch.tensor(self.data[idx][12:24], dtype=torch.float32)
        return input_frames, output_frames


def load_sevir(batch_size, val_batch_size, data_root, num_workers):
    dataset = SevirDataset(data_path=data_root + 'sevir_dataset.npy', transform=None)
    print(len(dataset))
    train_size = len(dataset) - 500
    test_size = 500

    train_dataset, test_dataset = data_utils.random_split(dataset, [train_size, test_size])
    
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_test, dataloader_test, mean, std


if __name__ == "__main__":
    data_path = '../../autodl-tmp/'
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_sevir(batch_size=16, 
                                                                                val_batch_size=16, 
                                                                                data_root=data_path, 
                                                                                num_workers=8)