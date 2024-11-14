import argparse

parser = argparse.ArgumentParser()
# Set-up parameters
parser.add_argument("--device", default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
parser.add_argument("--work_dir", default='./workdir', type=str)
parser.add_argument("--model", default='UNet', type=str)
parser.add_argument("--use_gpu", default=True, type=bool)
parser.add_argument("--debug", default=False, type=bool)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--checkpoint', default="", type=str)
parser.add_argument('--is_save_data', default=False, type=bool)

# dataset parameters
parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--data_root', default='../autodl-tmp/a2_fds.npy')   # ../autodl-tmp/ns_V1e-4_N10000_T30.mat or ./data/
parser.add_argument('--dataname', default='a2fds', choices=['mmnist', 'taxibj', 'caltech', 'navier'])
parser.add_argument('--num_workers', default=8, type=int)

# model parameters
# [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj -> (timesteps, in_channels, width, height)
# parser.add_argument('--in_shape', default=[4, 2, 32, 32], type=int, nargs='*')  10, 1, 64, 64
parser.add_argument('--in_shape', default=[50, 2, 32, 480], type=int, nargs='*')
parser.add_argument('--hid_S', default=64, type=int)
parser.add_argument('--hid_T', default=256, type=int)
parser.add_argument('--N_S', default=4, type=int)
parser.add_argument("--N_T", default=8, type=int)

parser.add_argument('--groups', default=4, type=int)
parser.add_argument('--res_units', default=32, type=int)
parser.add_argument('--res_layers', default=4, type=int)
parser.add_argument('--K', default=512, type=int)
parser.add_argument('--D', default=64, type=int)

# Training parameters
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--log_step', default=1, type=int)
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--load_pred_train', default=0, type=int, help='Learning rate')
parser.add_argument('--freeze_vqvae', default=0, type=int)
parser.add_argument('--theta', default=1, type=float)
args = parser.parse_args()


