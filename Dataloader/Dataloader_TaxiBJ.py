import torch
import numpy as np
from torch.utils.data import Dataset
from utils.dataloader import create_loader

class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        super(TrafficDataset, self).__init__()
        self.X = (X + 1) / 2
        self.Y = (Y + 1) / 2
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels


def load_taxibj(batch_size, val_batch_size, data_root, num_workers,  # OpenSTL
                pre_seq_length=None, aft_seq_length=None, in_shape=None,
                distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    dataset = np.load(data_root + 'taxibj/dataset.npz')
    # print(dataset.files)
    # taxibj数据集被 MinMax Normalization 缩放到了[-1, 1]范围
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
    print("X_train: ", X_train.shape, ", X_test: ", X_test.shape)
    train_set = TrafficDataset(X=X_train, Y=Y_train)
    test_set = TrafficDataset(X=X_test, Y=Y_test)


    # dataloader_train = create_loader(train_set,
    #                                  batch_size=batch_size,
    #                                  shuffle=True, is_training=True,
    #                                  pin_memory=True, drop_last=True,
    #                                  num_workers=num_workers,
    #                                  distributed=distributed, use_prefetcher=use_prefetcher)
    # dataloader_test = create_loader(test_set,
    #                                 batch_size=val_batch_size,
    #                                 shuffle=False, is_training=False,
    #                                 pin_memory=True, drop_last=drop_last,
    #                                 num_workers=num_workers,
    #                                 distributed=distributed, use_prefetcher=use_prefetcher)
    
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, None, dataloader_test, 0, 1