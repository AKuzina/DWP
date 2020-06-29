import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from nn.layers import *
from utils.kernel_dataset import *



class KernelVAE(pl.LightningModule):
    def __init__(self, args):
        """
        dim_latent --- dimention of the latent vector
        kernel --- size of the input kernel (7 or 5)
        """
        super(KernelVAE, self).__init__()
        dim_latent = args.dim_latent
        kernel = args.kernel_size

        self.warmup = args.warmup
        self.beta = 1/self.warmup
        self.lr = args.lr
        self.batch_size = args.batch_size

        path = os.path.join(args.root, args.dataset_name, 'no_prior/nb_-1')
        self.train_dset = KernelDataset(root=path, norm_thr=args.norm_thr,
                                        ker_size=args.kernel_size)
        print(len(self.train_dset))

        self.hid_channels = 128

        if kernel == 7:
            init_channels = 32
            self.encode = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ELU(inplace=True),
                nn.Conv2d(32, 64, 3),
                nn.ELU(inplace=True),
                nn.Conv2d(64, 64, 3),
                nn.ELU(inplace=True),
            )
            hid_ch = 64

            self.decode = nn.Sequential(
                nn.ConvTranspose2d(2, 64, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(64, 64, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(64, 32, 3),
                nn.ELU(inplace=True)
            )
            out_ch = 32
            
        elif kernel == 5:
            self.encode = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ELU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ELU(inplace=True),
                nn.Conv2d(64, 128, 3),
                nn.ELU(inplace=True),
                nn.Conv2d(128, 128, 3),
                nn.ELU(inplace=True)
            )
            hid_ch = 128

            self.decode = nn.Sequential(
                nn.Conv2d(dim_latent, 128, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 128, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 128, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 64, 1),
                nn.ELU(inplace=True)
            )
            out_ch = 64

        self.latent_mu = nn.Sequential(
            nn.Conv2d(hid_ch, dim_latent, 1),
            nn.Flatten()
        )
        self.latent_logsigma = nn.Sequential(
            nn.Conv2d(hid_ch, dim_latent, 1),
            nn.Flatten()
        )

        self.rec_mu = nn.Conv2d(out_ch, 1, 1)
        self.rec_logsigma = nn.Conv2d(out_ch, 1, 1)
        self.hid_shape = [-1, dim_latent, 1, 1]


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dset, self.batch_size,
                                           num_workers=0, shuffle=True)

    def encoder(self, x):
        x_enc = self.encode(x)
        latent_mu = self.latent_mu(x_enc)
        latent_logsigma = self.latent_logsigma(x_enc)
        return latent_mu, latent_logsigma

    def decoder(self, z):
        x_hat = self.decode(z.view(self.hid_shape))
        out_mu = self.rec_mu(x_hat)
        out_logsigma = self.rec_logsigma(x_hat)
        return out_mu, out_logsigma

    def gaussian_sampler(self, mu, logsigma):
        std = logsigma.exp().pow(0.5)
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std) + mu

    def forward(self, x):
        z_mu, z_logsigma = self.encoder(x)
        MB = x.size(0)
        N = 20
        z = torch.mean(
            self.gaussian_sampler(z_mu.view(MB, 1, -1).repeat(1, N, 1),
                                  z_logsigma.view(MB, 1, -1).repeat(1, N, 1)), dim=1)
        out_mu, out_logsigma = self.decoder(z)
        return out_mu, out_logsigma, z_mu, z_logsigma, z

    def loss(self, x, x_mu, x_logsigma, z_mu, z_logsigma, batch_idx):
        # print( x.shape, x_mu.shape, x_logsigma.shape, z_mu.shape, z_logsigma.shape)
        kl = -0.5 * (1 + z_logsigma - z_mu.pow(2) - z_logsigma.exp())
        kl = kl.sum(1, keepdim=True)

        re = 0.5*(x_logsigma + np.log(2 * np.pi) + (x_mu - x).pow(2) / x_logsigma.exp())
        re = re.view(re.shape[0], -1).sum(1, keepdim=True)

        # beta = torch.tensor(min(1., self.warmup*batch_idx))

        loss = torch.mean(re + self.beta * kl)

        return {'loss':loss, 'kl':kl.data.cpu().mean(), 're':re.data.cpu().mean(), 'beta':self.beta}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        X, y = batch
        x_mu, x_logsigma, z_mu, z_logsigma, z = self(X)
        logs = self.loss(X, x_mu, x_logsigma, z_mu, z_logsigma, batch_idx)
        self.beta = min(1., self.beta + 1/self.warmup)
        loss = logs.pop('loss')

        logs['train_loss'] = loss.data.cpu()
        return {'loss': loss, 'log': logs}


class KernelVAE3D(KernelVAE):
    def __init__(self, args):
        """
        dim_latent --- dimention of the latent vector
        kernel --- size of the input kernel
        """
        super(KernelVAE3D, self).__init__()
        if kernel == 3:
            ch = 32
            self.hid_channels = init_channels * 4

            self.encode = nn.Sequential(
                nn.Conv3d(1, ch, 3, padding=1),
                nn.MaxPool3d(2, padding=1),
                nn.ELU(inplace=True),
                nn.Conv3d(ch, ch * 2, 3, padding=1),
                nn.MaxPool3d(2),
                nn.ELU(inplace=True),
                nn.Conv3d(ch * 2, ch * 4, 1),
                nn.ELU(inplace=True),
                Flatten()
            )

            self.decode = nn.Sequential(
                nn.Conv3d(ch * 4, ch * 4, 3, padding=1),
                nn.ELU(inplace=True),
                nn.ConvTranspose3d(ch * 4, ch * 4, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose3d(ch * 4, ch * 2, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose3d(ch * 2, ch, 1),
                nn.ELU(inplace=True)
            )

        self.latent_mu = nn.Linear(self.hid_channels, dim_latent)
        self.latent_logsigma = nn.Linear(self.hid_channels, dim_latent)
        self.linear = nn.Linear(dim_latent, self.hid_channels)

        self.reconstruction_mu = nn.Sequential(
            nn.ConvTranspose3d(ch, 1, 1),
            nn.Tanh()
        )

        self.reconstruction_logsigma = nn.Sequential(
            nn.ConvTranspose3d(ch, 1, 1),
            nn.Tanh()
        )
        self.hid_shape = [-1, self.hid_channels, 1, 1, 1]
