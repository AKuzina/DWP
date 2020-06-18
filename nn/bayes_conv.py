import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple


class _BayesConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_BayesConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.mu_weight = nn.Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.logsigma_weight = nn.Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.mu_weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.logsigma_weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.logsigma_bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('logsigma_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.mu_weight.data.normal_(0, 0.02)
        self.logsigma_weight.data.fill_(-5)
        if self.mu_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.mu_bias, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.logsigma_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.logsigma_bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.mu_bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class BayesConv2d(_BayesConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, zero_mean=False,
                 threshold=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.zero_mean = zero_mean
        self.threshold = threshold
        self.log_alpha = None
        super(BayesConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, False, _pair(0), groups,
                                          bias)
        if zero_mean:
            self.mu_weight = nn.Parameter(torch.zeros_like(self.mu_weight))

    def forward(self, input):
        log_alpha = self.logsigma_weight - torch.log(self.mu_weight ** 2 + 1e-8)
        self.log_alpha = torch.clamp(log_alpha, -5, 5)
        if self.logsigma_bias is not None:
            bias = torch.pow(self.logsigma_bias, 2)
        else:
            bias = None

        if self.training:
            sigma_sq = F.conv2d(torch.pow(input, 2),
                                self.mu_weight ** 2 * torch.exp(self.log_alpha),
                                bias, self.stride, self.padding, self.dilation, self.groups)
            sigma_out = torch.sqrt(1e-4 + sigma_sq)

            mu_out = F.conv2d(input, self.mu_weight, self.mu_bias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            mask = (self.log_alpha < self.threshold).float()
            sigma_sq = F.conv2d(torch.pow(input, 2),
                                self.mu_weight ** 2 * torch.exp(self.log_alpha) * mask,
                                bias, self.stride, self.padding, self.dilation, self.groups)
            sigma_out = torch.sqrt(1e-4 + sigma_sq)

            mu_out = F.conv2d(input, self.mu_weight * mask, self.mu_bias, self.stride,
                              self.padding, self.dilation, self.groups)

        eps = sigma_out.data.new(sigma_out.size()).normal_()
        return eps.mul(sigma_out) + mu_out


class BayesConv3d(_BayesConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, zero_mean=False,
                 threshold=3):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        self.zero_mean = zero_mean
        self.threshold = threshold
        self.log_alpha = None

        super(BayesConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, False, _triple(0), groups,
                                          bias)
        if zero_mean:
            self.mu_weight = nn.Parameter(torch.zeros_like(self.mu_weight))

    def forward(self, input):
        log_alpha = self.logsigma_weight - torch.log(self.mu_weight ** 2 + 1e-8)
        self.log_alpha = torch.clamp(log_alpha, -5, 5)
        if self.logsigma_bias is not None:
            bias = torch.pow(self.logsigma_bias, 2)
        else:
            bias = None

        if self.training:
            sigma_sq = F.conv3d(torch.pow(input, 2),
                                self.mu_weight ** 2 * torch.exp(self.log_alpha), bias,
                                self.stride, self.padding, self.dilation, self.groups)
            sigma_out = torch.sqrt(1e-4 + sigma_sq)
            # if self.zero_mean:
            #     mu_out = torch.zeros_like(sigma_out)
            # else:
            mu_out = F.conv3d(input, self.mu_weight, self.mu_bias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            mask = (self.log_alpha < self.threshold).float()
            mu_out = F.conv3d(input, self.mu_weight * mask, self.mu_bias, self.stride,
                              self.padding,
                              self.dilation, self.groups)

            sigma_sq = F.conv3d(torch.pow(input, 2),
                                self.mu_weight ** 2 * torch.exp(self.log_alpha) * mask,
                                bias, self.stride, self.padding, self.dilation, self.groups)
            sigma_out = torch.sqrt(1e-4 + sigma_sq)

        eps = sigma_out.data.new(sigma_out.size()).normal_()
        return eps.mul(sigma_out) + mu_out