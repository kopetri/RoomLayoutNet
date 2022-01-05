import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from nested_unet import NestedUNet as Unet

class LayoutSegmentation(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = Unet(1, input_channels=3)
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        X,Y = batch
        Y_hat = self.model(X)
        loss = self.criterion(Y_hat, Y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X,Y = batch
        Y_hat = self.model(X)
        loss = self.criterion(Y_hat, Y)
        self.log("valid_loss", loss, prog_bar=True)
        return {'valid_loss': loss}
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]