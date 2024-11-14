import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

class Fire_dataset(Dataset):
    def __init__(self, data_paths, transform=None):
        data_list = [np.load(path) for path in data_paths]
        self.data = np.concatenate(data_list, axis=0)
        self.data = torch.from_numpy(self.data).float()  
        self.transform = transform

        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

        # self.mean = 0
        # self.std = 1
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames = self.data[idx][:50]
        output_frames = self.data[idx][50:]
        if self.transform:
            input_frames = self.transform(input_frames)
            output_frames = self.transform(output_frames)

        input_frames = (input_frames - self.mean) / self.std
        output_frames = (output_frames - self.mean) / self.std

        return input_frames, output_frames


import os

# folder_path = '/data/workspace/yancheng/fire_net/fire_solver/dataset_fds/'


def load_file(batch_size, val_batch_size, data_root, num_workers):
    npy_files = [data_root]
    npy_files.sort()
    full_dataset = Fire_dataset(npy_files)
    print(len(full_dataset))
    mean, std = full_dataset.mean, full_dataset.std
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    if isinstance(train_dataset, torch.utils.data.Subset):
        train_dataset.dataset.mean = mean
        train_dataset.dataset.std = std

    if isinstance(test_dataset, torch.utils.data.Subset):
        test_dataset.dataset.mean = mean
        test_dataset.dataset.std = std

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

#     val_size = int(0.1 * train_size)
#     train_size_new = train_size - val_size
#     train_dataset, val_dataset = random_split(train_dataset, [train_size_new, val_size])

#     if isinstance(val_dataset, torch.utils.data.Subset):
#         val_dataset.dataset.mean = mean
#         val_dataset.dataset.std = std

#     dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
#     dataloader_validation = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_test, dataloader_test, mean, std

