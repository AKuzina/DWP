import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.bayes_conv import BayesConv2d


class MNISTnet(nn.Module):
    def __init__(self, n_classes=10, bayes=False, priors=None):
        super(MNISTnet, self).__init__()

        n_channels = [256, 512]
        conv = nn.Conv2d
        if bayes:
            conv = BayesConv2d
            self.priors = priors
            if priors is None:
                self.priors = []
        self.net = nn.Sequential(
            conv(1, n_channels[0], 7),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            conv(n_channels[0], n_channels[1], 5),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten())
        self.clf = nn.Linear(4608, n_classes)

    def forward(self, x):
        f = self.net(x)
        return self.clf(f)