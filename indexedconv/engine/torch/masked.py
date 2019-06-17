"""
masked.py
========
Contain the core functions for the masked operations
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

from indexedconv.utils import build_kernel


class MaskedConv2d(nn.Conv2d):

    def forward(self, x):
        r = int((self.kernel_size[0] - 1) / 2)
        mask = torch.from_numpy(build_kernel('Hex', radius=r)).type(torch.float).to(x.device)
        return F.conv2d(x, self.weight * mask, bias=self.bias, stride=self.stride, padding=self.padding)
