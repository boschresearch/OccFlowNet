# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from ..builder import LOSSES
import torch
import torch.nn as nn

def silog_loss(pred, target, lambd=.85):
    d = torch.log(pred + 1e-7) - torch.log(target)
    return torch.sqrt((d ** 2).mean() - lambd * (d.mean() ** 2)) 
    # return torch.sqrt((d ** 2).mean() - lambd * (d.mean() ** 2)) * 10.0

@LOSSES.register_module()
class SiLogLoss(nn.Module):
    def __init__(self, loss_weight=1.0, lambd=.85, loss_name='loss_silog'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name 
        self.lambd = lambd
    def forward(self, pred, target, **kwargs):
        return self.loss_weight * silog_loss(pred, target, lambd=self.lambd)