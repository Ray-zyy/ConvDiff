from utils.dataloader_taxibj import load_taxibj
from utils.dataloader_moving_mnist import load_moving_mnist
from utils.dataloader_navier import load_navier
from utils.dataloader_sevir import load_sevir
from utils.dataloader_navierv1 import load_navierv1
from utils.dataloader_file import load_file

def load_data(dataname, batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_moving_mnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'navier':
        return load_navier(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'navierT30':
        return load_navierv1(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'navierT50':
        return load_navierv1(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'sevir':
        return load_sevir(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'a1fds':
        return load_file(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'a2fds':
        return load_file(batch_size, val_batch_size, data_root, num_workers)
