import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from nested_unet import NestedUNet as Unet
from utils import extranct_coors
import wandb

class LayoutSegmentation(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
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

        if batch_idx in np.arange(0,5,1):
            pred, gt, edges, corn = extranct_coors(Y_hat, Y)
            self.logger.experiment.log({"validation_batch_{}".format(batch_idx):[
                wandb.Image(pred, caption="valid_prediction"),
                wandb.Image(gt, caption="valid_ground_truth"),
                wandb.Image(edges, caption="valid_edges"),
                wandb.Image(corn, caption="valid_corners")
                ]})
        
        self.log("valid_loss", loss, prog_bar=True)
        return {'valid_loss': loss}

    def test_step(self, batch, batch_idx):
        X,Y = batch
        Y_hat = self.model(X)
        loss = self.criterion(Y_hat, Y)
        self.log("test_loss", loss, prog_bar=True)
        return {'test_loss': loss}
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]