# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from ..builder import LOSSES
import torch.nn as nn

@LOSSES.register_module()
class HuberLoss(nn.Module):
    def __init__(self, loss_weight=1.0, delta=1.0, loss_name='loss_huber'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.loss_fn = nn.HuberLoss(delta=delta)

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * self.loss_fn(pred, target)