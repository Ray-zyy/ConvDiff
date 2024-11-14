import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import scipy.io


class NavierDataset(Dataset):
    def __init__(self, data_path, transform=None):
        print(data_path)
        data = scipy.io.loadmat(data_path)['u']
        data = data.transpose(0, 3, 1, 2)
        self.data = np.expand_dims(data, axis=2)
        print(self.data.shape)
        self.transform = transform
        # self.mean = np.mean(self.data)
        # self.std = np.std(self.data)
        # self.data = (self.data - self.mean) / self.std

        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames = torch.tensor(self.data[idx][:10], dtype=torch.float32)
        output_frames = torch.tensor(self.data[idx][10:], dtype=torch.float32)
        return input_frames, output_frames



def load_navier(batch_size, val_batch_size, data_root, num_workers):
    dataset = NavierDataset(data_path=data_root, transform=None)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = data_utils.random_split(dataset, [train_size, test_size])
    
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_test, dataloader_test, mean, std



# if __name__ == "__main__":
#     data_path = '../data/NavierStokes_V1e-5_N1200_T20.mat'
#     dataloader_train, dataloader_validation, dataloader_test, mean, std = load_navier(batch_size=16, 
#                                                                             val_batch_size=16, 
#                                                                             data_root=data_path, 
#                                                                             num_workers=8)