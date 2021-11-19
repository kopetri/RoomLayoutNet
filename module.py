import torch
from math import log10
import pytorch_lightning as pl
from torchvision.utils import make_grid
from pix2pix import get_scheduler, GANLoss, define_D, define_G
from model import HorizonNet
import numpy as np
import torch.nn.functional as F
from inference import inference
from eval_general import test_general

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clone().detach()
        tensor = tensor.squeeze(0).permute(1,2,0)
        tensor = tensor.cpu().numpy().astype(np.float32)
    return tensor

def nan_check(tensor):
    return not torch.any(torch.isnan(tensor))

class LayoutEstimationImprover(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.generator = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, 'batch', False, 'normal', 0.02)
        self.discriminator = define_D(self.opt.input_nc + self.opt.output_nc, self.opt.ndf, self.opt.netD)
        self.layout_estimator = HorizonNet('resnet50', use_rnn=True, gan_c=1)
        self.criterionGAN = GANLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.log_step = 0

    def forward(self, x):
        # forward
        fake = self.generator(x)
        return fake
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        rgb, y_bon, y_cor, mask = batch
        fake = self(rgb)

        if batch_idx % 100 == 0:
            tensorboard = self.logger.experiment
            row_0 = make_grid(  rgb, nrow=self.opt.batch_size) 
            row_1 = make_grid( mask, nrow=self.opt.batch_size)
            row_2 = make_grid( fake, nrow=self.opt.batch_size)
            grid = torch.cat([row_0, row_1, row_2], dim=1)
            tensorboard.add_image('viz', grid, self.log_step)
            self.log_step += 1

        if optimizer_idx == 0:
            ######################
            # (1) Update D network
            ######################
            
            # train with fake
            fake_ab = torch.cat((rgb, fake), 1)
            pred_fake = self.discriminator.forward(fake_ab.detach())
            loss_d_fake = self.criterionGAN(pred_fake, False)

            # train with real
            real_ab = torch.cat((rgb, mask), 1)
            pred_real = self.discriminator.forward(real_ab)
            loss_d_real = self.criterionGAN(pred_real, True)
            
            # Combined D loss
            d_loss = (loss_d_fake + loss_d_real) * 0.5
            
            self.log('d_loss', d_loss)
            return {'loss': d_loss}

        elif optimizer_idx == 1:
            ######################
            # (2) Update G network
            ######################

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((rgb, fake), 1)
            pred_fake = self.discriminator.forward(fake_ab)
            loss_g_gan = self.criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = self.criterionL1(fake, mask) * self.opt.lamb
            
            g_loss = loss_g_gan + loss_g_l1
            mse = self.criterionMSE(fake.detach(), mask.detach())
            self.log('g_loss', g_loss)
            self.log('train_mse', mse, prog_bar=True, on_epoch=True)
            return {'loss': g_loss, "train_mse": mse}

        elif optimizer_idx == 2:
            y_bon_, y_cor_ = self.layout_estimator(rgb, fake)
            bon_loss = F.l1_loss(y_bon_, y_bon)
            cor_loss = F.binary_cross_entropy_with_logits(y_cor_, y_cor)
            self.log('bon_loss', bon_loss, prog_bar=True, on_epoch=True)
            self.log('cor_loss', cor_loss, prog_bar=True, on_epoch=True)
            return {'loss': bon_loss + cor_loss}

    def validation_step(self, batch, batch_idx):
        x, y_bon, y_cor, gt_cor_id = batch   

        features_g = self(x)     
        
        losses = {}

        y_bon_, y_cor_ = self.layout_estimator(x, features_g)
        losses['bon'] = F.l1_loss(y_bon_, y_bon)
        losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor)
        losses['total'] = losses['bon'] + losses['cor']

        # True eval result instead of training objective
        true_eval = dict([
            (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
            for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
        ])
        try:
            dt_cor_id = inference(self.net, x, x.device, force_raw=True)[0]
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
            

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        optimizer_l = torch.optim.Adam(self.layout_estimator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        scheduler_g = get_scheduler(optimizer_g, self.opt)
        scheduler_d = get_scheduler(optimizer_d, self.opt)
        scheduler_l = get_scheduler(optimizer_l, self.opt)
        return [optimizer_d, optimizer_g, optimizer_l], [scheduler_d, scheduler_g, scheduler_l]