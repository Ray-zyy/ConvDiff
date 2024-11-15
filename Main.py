import os
import torch
from config import args
from utils.utils import *
from utils.data_preparation import load_data
from engine import Engine
# from models.SimVP import SimVP
from models.simvp import SimVP
from models.ConvLSTM import ConvLSTM
from models.UNet import UNet
from models.eartherformer_model import CuboidTransformerModel
from models.model4 import Convdiff
# from models.Convdiff import Convdiff
# from models.STResNet import ST_ResNet

if __name__ == '__main__':
    # 1.实验环境
    if args.use_gpu:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: [cuda:{}]'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print("Use CPU")
    # 2.固定随机数种子，创建日志/模型目录
    init_seed(args.seed)
    args.log_dir = get_log_dir(args.model, args.dataname)
    args.checkpoint = os.path.join(args.log_dir, '{}_{}_best_model.pth'.format(args.dataname, args.model))
    logger = None
    if os.path.isdir(args.log_dir) == False:
        os.makedirs(args.log_dir, exist_ok=True)
        logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        logger.info("Experiment log path in: {}".format(args.log_dir))
        logger.info(args.checkpoint)
    # 3.加载数据
    config = args.__dict__
    train_loader, valid_loader, test_loader, data_mean, data_std = load_data(**config)
    valid_loader = test_loader if valid_loader is None else valid_loader

    # 4.初始化组件(model, optimizer, criterion)
    model = Convdiff(tuple(args.in_shape), 
                   args.hid_S,
                   args.hid_T, 
                   args.N_S, 
                   args.N_T).to(device)
    
    # model = UNet(n_channels=50 * 2, out_channels=50 * 2, bilinear=True).to(device)
    # model = ST_ResNet(tuple(args.in_shape)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs
    )
    criterion = torch.nn.MSELoss()
    exp = Engine(
        args, 
        train_loader, 
        valid_loader, 
        test_loader, 
        (data_mean, data_std),
        model, 
        optimizer, 
        scheduler, 
        criterion,
        logger, 
        device
    )
    # 5.模型训练/测试
    if not args.debug:
        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>> start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.train()
        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>> test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test()
    else:
        # 模型保存路径
        args.checkpoint = "输入保持路径"
        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>> test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test()
