import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.layers import *


class KernelVAE3D(nn.Module):
    def __init__(self, dim_latent=2, kernel=3, non_random=False):
        """
        dim_latent --- dimention of the latent vector
        kernel --- size of the input kernel
        """
        super(KernelVAE3D).__init__()
        self.non_random = non_random
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

    def encoder(self, x):
        # encode
        x_enc = self.encode(x)
        latent_mu = self.latent_mu(x_enc)
        latent_logsigma = self.latent_logsigma(x_enc)
        return latent_mu, latent_logsigma

    def decoder(self, z):
        z = self.linear(z)
        x_hat = self.decode(z.view(-1, self.hid_channels, 1, 1, 1))
        out_mu = self.reconstruction_mu(x_hat)
        out_logsigma = self.reconstruction_logsigma(x_hat)
        return out_mu, out_logsigma

    def gaussian_sampler(self, mu, logsigma):
        if self.non_random:
            return mu
        else:
            std = torch.exp(logsigma).pow(0.5)
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std) + mu

    def forward(self, x):
        latent_mu, latent_logsigma = self.encoder(x)
        N = 20
        h_hid = torch.mean(
            self.gaussian_sampler(latent_mu.view(x.size(0), 1, -1).repeat(1, N, 1),
                                  latent_logsigma.view(x.size(0), 1, -1).repeat(1, N, 1)),
            dim=1)
        out_mu, out_logsigma = self.decoder(h_hid)
        return out_mu, out_logsigma, latent_mu, latent_logsigma, h_hid

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage,
                                                                loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class KernelVAE(nn.Module):
    def __init__(self, dim_latent=2, kernel=7):
        """
        dim_latent --- dimention of the latent vector
        kernel --- size of the input kernel (7 or 5)
        """
        super(KernelVAE).__init__()

        if kernel == 7:
            self.hid_channels = 128
            init_channels = 64
            self.encode = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.MaxPool2d(2),
                nn.ELU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.MaxPool2d(2),
                nn.ELU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ELU(inplace=True),
                Flatten()
            )

            self.decode = nn.Sequential(
                nn.ConvTranspose2d(128, 128, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 128, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 64, 3),
                nn.ELU(inplace=True)
            )
        elif kernel == 5:
            self.hid_channels = 128
            init_channels = 64
            self.encode = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.MaxPool2d(2),
                nn.ELU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.MaxPool2d(2),
                nn.ELU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ELU(inplace=True),
                Flatten()
            )

            self.decode = nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 128, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 128, 3),
                nn.ELU(inplace=True),
                nn.ConvTranspose2d(128, 64, 1),
                nn.ELU(inplace=True)
            )

        self.latent_mu = nn.Linear(self.hid_channels, dim_latent)
        self.latent_logsigma = nn.Linear(self.hid_channels, dim_latent)
        self.linear = nn.Linear(dim_latent, self.hid_channels)
        self.tanh = nn.Tanh()

        self.reconstruction_mu = nn.Sequential(
            nn.ConvTranspose2d(init_channels, 1, 1),
            nn.Tanh()
        )

        self.reconstruction_logsigma = nn.Sequential(
            nn.ConvTranspose2d(init_channels, 1, 1),
            nn.Tanh()
        )

    def encoder(self, x):
        # encode
        x_enc = self.encode(x)
        latent_mu = self.latent_mu(x_enc)
        latent_logsigma = self.latent_logsigma(x_enc)
        return latent_mu, latent_logsigma

    def decoder(self, z):
        z = self.linear(z)
        x_hat = self.decode(z.view(-1, self.hid_channels, 1, 1))
        out_mu = self.reconstruction_mu(x_hat)
        out_logsigma = self.reconstruction_logsigma(x_hat)
        return out_mu, out_logsigma

    def gaussian_sampler(self, mu, logsigma):
        std = logsigma.exp().pow(0.5)
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std) + mu

    def forward(self, x):
        latent_mu, latent_logsigma = self.encoder(x)

        h_hid = torch.mean(
            self.gaussian_sampler(latent_mu.view(x.size(0), 1, -1).repeat(1, 5, 1),
                                  latent_logsigma.view(x.size(0), 1, -1).repeat(1, 5, 1)),
            dim=1)
        out_mu, out_logsigma = self.decoder(h_hid)
        return out_mu, out_logsigma, latent_mu, latent_logsigma, h_hid

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage,
                                                                loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')