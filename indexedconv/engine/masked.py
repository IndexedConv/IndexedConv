"""
masked.py
========
Contain the core functions for the masked operations
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):

    def forward(self, x):
        if self.kernel_size == (3, 3):
            mask = torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]]).to(x.device)
        elif self.kernel_size == (5, 5):
            mask = torch.tensor([[1., 1., 1., 0., 0.],
                                 [1., 1., 1., 1., 0.],
                                 [1., 1., 1., 1., 1.],
                                 [0., 1., 1., 1., 1.],
                                 [0., 0., 1., 1., 1.]]).to(x.device)
        return F.conv2d(x, self.weight * mask, bias=self.bias, stride=self.stride, padding=self.padding)
