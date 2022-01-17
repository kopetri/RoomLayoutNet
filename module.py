import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from nested_unet import NestedUNet as Unet
from utils import extranct_coors
import wandb
from bts import BtsModel
from metrics import Delta, IoU, extract_corners, corner_error
from torchvision.utils import save_image
from pathlib import Path
from inference import inference
from eval_general import test_general
from eval_cuboid import test

def loss_function(loss):
    if loss == 'mse':
        return nn.MSELoss()
    elif loss == 'mae':
        return nn.L1Loss()
    else:
        raise NotImplementedError("unknown loss: ", loss)

class LayoutSegmentation(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        #self.model = Unet(1, input_channels=3)
        self.model = BtsModel(opt)
        self.criterion = loss_function(opt.loss)
        self.iou3 = IoU(0.9)
        self.iou2 = IoU(0.99)
        self.iou1 = IoU(0.999)

    def forward(self, X):
        _, _, _, _, Y_hat = self.model(X)
        return Y_hat

    def training_step(self, batch, batch_idx):
        X,Y = batch['img'], batch['dist']
        Y_hat = self(X)
        loss = self.criterion(Y_hat, Y)
        iou1 = self.iou1(Y_hat, Y)
        iou2 = self.iou2(Y_hat, Y)
        iou3 = self.iou3(Y_hat, Y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou1", iou1, prog_bar=True)
        self.log("train_iou2", iou2, prog_bar=True)
        self.log("train_iou3", iou3, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X,Y, gt_cor_id = batch['img'], batch['dist'], batch['cor'][0]
        Y_hat = self(X)
                
        if batch_idx in np.arange(0,5,1):
            #pred, gt, edges, corn = extranct_coors(Y_hat, Y)
            self.logger.experiment.log({"validation_batch_{}".format(batch_idx):[
                wandb.Image(Y_hat, caption="valid_prediction"),
                wandb.Image(Y, caption="valid_ground_truth"),
                wandb.Image(X, caption="valid_img")
                ]})
        true_eval = dict([
            (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
            for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
        ])
        losses = {}
        try:
            N = len(gt_cor_id)
            dt_cor_id = extract_corners(Y_hat, N=N)
            dt_cor_id[:, 0] *= 1024
            dt_cor_id[:, 1] *= 512
        except Exception as e:
            dt_cor_id = np.array([
                [k//2 * 1024, 256 - ((k%2)*2 - 1) * 120]
                for k in range(8)
            ])
        test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)
        losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
        losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
        losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
        losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']])

        self.log("valid_2DIoU",   losses['2DIoU'], prog_bar=True)
        self.log("valid_3DIoU",   losses['3DIoU'], prog_bar=True)
        self.log("valid_rmse",    losses['rmse'], prog_bar=True)
        self.log("valid_delta_1", losses['delta_1'], prog_bar=True)

        return {'valid_2DIoU': losses['2DIoU'], 'valid_3DIoU': losses['3DIoU'], 'valid_rmse': losses['rmse'], 'valid_delta_1': losses['delta_1']}

    def test_step(self, batch, batch_idx):
        X,Y = batch['img'], batch['dist']
        Y_hat = self(X)
        loss = self.criterion(Y_hat, Y)
        iou1 = self.iou1(Y_hat, Y)
        iou2 = self.iou2(Y_hat, Y)
        iou3 = self.iou3(Y_hat, Y)
        
        corners_gt = extract_corners(Y, N=8)
        corners_id = extract_corners(Y_hat, N=8)
        ce = corner_error(corners_gt, corners_id, 512, 256)
        
        
        self.log("loss", loss, prog_bar=True)
        self.log("ce", ce, prog_bar=True)
        self.log("iou1", iou1, prog_bar=True)
        self.log("iou2", iou2, prog_bar=True)
        self.log("iou3", iou3, prog_bar=True)
        return {'loss': loss, 'iou1': iou1, 'iou2':iou2, 'iou3':iou3, 'ce': ce}
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]