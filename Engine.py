import os
import torch
from tqdm import tqdm
from utils.metrics import *
from utils.recorder import *

class Engine(object):
    def __init__(self, args, train_loader, valid_loader, test_loader, scaler, model, optimizer, scheduler, criterion, logger, device):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.mean, self.std = scaler[0], scaler[1]
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.logger = logger
        self.device = device

    def train(self):
        recorder = Recorder(verbose=True)
        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()
            train_pbar = self.train_loader

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                print(pred_y.shape)
                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            
            train_loss = np.average(train_loss)

            if epoch % self.args.log_step == 0:
                with torch.no_grad():
                    val_loss = self.valid()
                self.logger.info("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}.".format(epoch + 1, train_loss, val_loss))
                recorder(val_loss, self.model, self.logger, self.args.checkpoint)
        # 读取模型文件
        # self.model.load_state_dict(torch.load(self.args.checkpoint))
        # return self.model


    def valid(self):
        self.model.eval()
        preds_list, trues_list, total_loss = [], [], []
        valid_pbar = self.valid_loader
        for i, (batch_x, batch_y) in enumerate(valid_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            preds_list.append(pred_y.detach().cpu().numpy())
            trues_list.append(batch_y.detach().cpu().numpy())
            loss = self.criterion(pred_y, batch_y)
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, 0, 1, True)
        self.logger.info('Valid MSE:{:.4f}, MAE:{:.4f}, SSIM:{:.4f}, PSNR:{:.4f}.'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss


    def test(self):
        self.model.load_state_dict(torch.load(self.args.checkpoint))
        self.model.eval()
        inputs_list, trues_list, preds_list = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))

            inputs_list.append(batch_x.detach().cpu().numpy())
            trues_list.append(batch_y.detach().cpu().numpy())
            preds_list.append(pred_y.detach().cpu().numpy())
            
        inputs, trues, preds = map(lambda data: np.concatenate(data, axis=0), [inputs_list, trues_list, preds_list])
        mse, mae, ssim, psnr = metric(preds, trues, 0, 1, True)
        self.logger.info(('Test MSE:{:.4f}, MAE:{:.4f}, SSIM:{:.4f}, PSNR:{:.4f}'.format(mse, mae, ssim, psnr)))
        if self.args.is_save_data:
            for np_data in ['inputs', 'trues', 'preds']:
                np.save(os.path.join(self.args.log_dir, np_data + '.npy'), vars()[np_data])

