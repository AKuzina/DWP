import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.bayes_conv import BayesConv2d


class MNISTnet(nn.Module):
    def __init__(self, n_classes=10, bayes=False):
        super(MNISTnet, self).__init__()

        n_channels = [256, 512, 128]
        conv = nn.Conv2d
        if bayes:
            conv = BayesConv2d
        self.net = nn.Sequential(
            conv(1, n_channels[0], 7),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            conv(n_channels[0], n_channels[1], 5),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4608, n_classes))

    def forward(self, x):
        return self.net(x)