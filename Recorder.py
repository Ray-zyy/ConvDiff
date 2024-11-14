import torch
import numpy as np

class Recorder:
    def __init__(self, verbose=False):
        self.verbose = verbose
        # 设置初始的验证误差是无穷大
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model, logger, checkpoint):
        if val_loss < self.val_loss_min:
            self.save_checkpoint(val_loss, model, logger, checkpoint)

    def save_checkpoint(self, val_loss, model, logger, checkpoint):
        if self.verbose:
            logger.info(f'Validation loss decreased {self.val_loss_min:.6f} --> {val_loss:.6f}, the best model saved!\n')
        torch.save(model.state_dict(), checkpoint)
        self.val_loss_min = val_loss
