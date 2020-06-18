import torch
import torch.nn as nn
import pytorch_lightning as pl

import models
import utils
import nn.loss as losses
import nn.metrics as metr

class BaseModel(pl.LightningModule):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.train_dset, self.test_dset, args = utils.load_dataset(args)
        self.net, args = models.init_net(args)
        
        self.batch_size = args.batch_size
        self.lr = args.lr
        if args.task == 'clf':
            self.loss_fun = losses.cr_ent_ll
            self.metrics = {'Accuracy': metr.accuracy}
        elif args.task == 'seg':
            self.loss_fun = losses.unet_binary_ll
            self.metrics = {'IOU':metr.iou, 'DICE':metr.dice}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dset, self.batch_size,
                                           num_workers=4, shuffle=True)

    def validation_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dset, self.batch_size,
                                           num_workers=4, shuffle=False)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss_fun(out, y)

        logs = {'train_loss':loss}
        _, y_pred = torch.max(out.detach(), 1)
        for m in self.metrics:
            logs[m] = self.metrics[m](y.cpu().numpy(),
                                      y_pred.cpu().numpy())
        return {'loss':loss, 'log':logs}
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss_fun(out, y)

        logs = {'validation_loss': loss}
        _, y_pred = torch.max(out.detach(), 1)
        for m in self.metrics:
            logs[m] = self.metrics[m](y.cpu().numpy(), y_pred.cpu().numpy())
        return {'loss': loss, 'log': logs}
        
# class UNetDWP(DWP, UNet3D):
#     """
#     Forward from unet
#
#     """
#     def __init__(self):
#
#     def prepare_data(self):
#         return NotImplemented