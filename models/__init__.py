import os

import torch
import torch.nn as nn
import torch.optim as optim

from .unet import UNet3D
from .kernel_vae import KernelVAE3D, KernelVAE
from .clf_net import MNISTnet
from .dwp import BaseModel



# unet_layers = [
# ['init_conv.weight', 'down1.conv_1.conv.2.weight'],  # 1
# ['down1.conv_2.conv.2.weight', 'down2.conv_1.conv.2.weight',
#  'down2.conv_2.conv.2.weight', 'down3.conv_1.conv.2.weight'],  # 2
# ['down3.conv_2.conv.2.weight', 'down4.conv_1.conv.2.weight',
#  'down4.conv_2.conv.2.weight', 'down5.conv_1.conv.2.weight'],  # 3
# ['down5.conv_2.conv.2.weight', 'down6.conv_1.conv.2.weight', 'down6.conv_2.conv.2.weight',
#  'down7.conv_1.conv.2.weight', 'down7.conv_2.conv.2.weight','down8.conv_1.conv.2.weight',
#  'down8.conv_2.conv.2.weight', 'down9.conv_1.conv.2.weight', 'down9.conv_2.conv.2.weight'],  # 4
# ['up1.conv_1.conv.2.weight', 'up1.conv_1.conv.2.weight'],  # 5
# ['up2.conv_1.conv.2.weight', 'up2.conv_1.conv.2.weight'],  # 6
# ['up3.conv_1.conv.2.weight', 'up3.conv_1.conv.2.weight']]  # 7
#
# clf_layers = [
# ['init_conv.weight', 'encoder.0.conv_1.conv.2.weight'],  # 1
# ['encoder.0.conv_2.conv.2.weight', 'encoder.1.conv_1.conv.2.weight',
#  'encoder.1.conv_2.conv.2.weight', 'encoder.2.conv_1.conv.2.weight'],  # 2
# ['encoder.2.conv_2.conv.2.weight', 'encoder.3.conv_1.conv.2.weight',
#  'encoder.3.conv_2.conv.2.weight', 'encoder.4.conv_1.conv.2.weight'],  # 3
# ['encoder.4.conv_2.conv.2.weight', 'encoder.5.conv_1.conv.2.weight', 'encoder.5.conv_2.conv.2.weight',
#  'encoder.6.conv_1.conv.2.weight', 'encoder.6.conv_2.conv.2.weight','encoder.7.conv_1.conv.2.weight',
#  'encoder.7.conv_2.conv.2.weight', 'encoder.8.conv_1.conv.2.weight', 'encoder.8.conv_2.conv.2.weight']]  # 4


def init_net(args):
    if args.short:
        args.n_channels = [1, 16, 32, 32, 64]
        args.down_layers = 7
    else:
        args.n_channels = [1, 16, 32, 64, 128]
        args.down_layers = 10

    if 'clf' in args.task:
        if 'MNIST' in args.dataset_name:
            net = MNISTnet(n_classes=10, bayes=args.dwp)
    elif 'seg' in args.task:
        net = UNet3D(args.n_classes, n_channels=args.n_channels, bayes=args.dwp,
                     shorten=args.short, devices=args.devices)
    # net = net_to_device(args, net)
    return net, args


# def net_to_device(args, net):
#     if args.devices is not None:
#         net.init_conv.to(args.devices[0])
#         if 'seg' in args.task:
#             for i in range(1, args.down_layers):
#                 getattr(net, 'down{}'.format(i)).to(args.devices[0])
#             for i in range(1, 4):
#                 getattr(net, 'up{}'.format(i)).to(args.devices[1])
#             net.out.to(args.devices[1])
#
#         elif 'clf' in args.task:
#             net.encoder.to(args.devices[0])
#             net.classifier.to(args.devices[1])
#     else:
#         net.to(args.device)
#     return net


def init_dwp(args):
    args.net_priors = []
    args.opt_priors = []
    root = '/'.join(args.root.split('/')[:-1])
    weights = os.path.join(root, 'runs/weights/{}/dwp/'.format(args.prior))

    if 'clf' in args.task:
        args.al = clf_layers
    else:
        args.al = unet_layers
    if args.short:
        args.al[3] = args.al[3][:3]

    for i in range(len(args.al)):
        args.net_priors.append(Kernel_3D_VAE(dim_latent = 6, kernel = 3))
        args.net_priors[i].load_weights(os.path.join(weights, 'DWPVAE_layer{}_0301/best_model.pth'.format(i+1)))
        if args.devices is not None:
            args.net_priors[i].to(args.devices[1])
        else:
            args.net_priors[i].to(args.device)
        for param in args.net_priors[i].decode.parameters():
            param.requires_grad = False
        for param in args.net_priors[i].linear.parameters():
            param.requires_grad = False
        for param in args.net_priors[i].reconstruction_mu.parameters():
            param.requires_grad = False
        for param in args.net_priors[i].reconstruction_logsigma.parameters():
            param.requires_grad = False
        args.opt_priors.append(optim.Adam(filter(lambda p: p.requires_grad, args.net_priors[i].parameters())))
    return args