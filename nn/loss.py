import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from itertools import repeat
import numpy as np


def dice_loss(input, target):
    eps = 0.001
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    union = iflat.sum() + tflat.sum()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + eps) / (union + eps))

def log_norm(x, mu, logsigma):
    out = - 0.5 * (logsigma + (mu - x).pow(2) / logsigma.exp())
    return out.sum()

def cr_ent_ll(y_pred, y_true):
    cr_en = F.cross_entropy(y_pred, y_true)
    return {'ll':cr_en}

def unet_binary_ll(y_pred, y_true, dice_weight=0.99):
    cr_en = F.cross_entropy(y_pred, y_true)
    dice = dice_loss(F.softmax(y_pred, dim=1)[:, 1], y_true.type_as(y_pred))

    loss = (1-dice_weight)*cr_en + dice_weight*dice
    return {'ll':loss, 'Cross-entropy':cr_en, 'Dice':dice}


class BayesLoss:
    def __init__(self, log_lik=cr_ent_ll, anneal=None):
        self.log_lik = log_lik
        self.anneal = anneal
        self.w = 1
        if self.anneal is not None:
            self.w = self.anneal

    def approx_KL(self, *kwargs):
        return NotImplementedError

    def __call__(self, y_pred, y_true, KL_appprox):
        stats = self.log_lik(y_pred, y_true)
        stats['KL'] = KL_appprox

        loss = stats['ll'] + self.w * KL_appprox
        if self.anneal is not None:
            self.w += self.anneal
        return loss, stats


class DWPLoss(BayesLoss):
    def __init__(self, log_lik=cr_ent_ll, anneal=None):
        super(DWPLoss, self).__init__(log_lik=log_lik, anneal=anneal)

    def approx_KL(self, w, w_mu, w_logsigma, z, z_mu, z_logsigma, rec_w_mu, rec_w_logsigma):
        # approximate posterior on weights
        log_q = log_norm(w, w_mu, w_logsigma)
        # encoder of dwp
        log_r = log_norm(z, z_mu, z_logsigma)
        # decoder of dwp
        log_pw_z = log_norm(w, rec_w_mu, rec_w_logsigma)
        # prior on latent space in dwp
        log_pz = - 0.5 * z.pow(2).sum()
        return log_q + log_r - log_pz - log_pw_z


class VarDropoutLoss(BayesLoss):
    def __init__(self, log_lik=cr_ent_ll, anneal=None):
        super(VarDropoutLoss, self).__init__(log_lik=log_lik, anneal=anneal)

    def approx_KL(self, mu, logsigma):
        out = 0.5 * (-1 - logsigma + mu.pow(2) + logsigma.exp())
        return out.sum()
