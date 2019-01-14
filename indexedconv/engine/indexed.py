"""
indexed.py
========
Contain the core functions for the indexed operations
"""

import logging
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import indexedconv.utils as utils


class IndexedMaxPool2d(nn.Module):
    r"""
    Compute the Max Pooling 2d operation on a batch of features of vector images wrt a matrix of indices

    Args:
        indices (LongTensor): index tensor of shape (L x kernel_size),
            having on each row the indices of neighbors of each element of the input
            a -1 indicates the absence of a neighbor, which is handled as zero-padding
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`
    """
    def __init__(self, indices):
        super(IndexedMaxPool2d, self).__init__()
        self.logger = logging.getLogger(__name__ + '.IndexedMaxPool2d')
        self.indices = indices
        self.indices, self.mask = utils.prepare_mask(self.indices)

    def forward(self, input_images):
        self.logger.debug('Max pool image')

        self.indices = self.indices.to(input_images.device)
        self.mask = self.mask.to(input_images.device)

        col = input_images[..., self.indices] * self.mask

        out, _ = torch.max(col, 2)

        return out


class IndexedAveragePool2d(nn.Module):
    r"""
    Compute the Average Pooling 2d operation on a batch of features of vector images wrt a matrix of indices

    Args:
        indices (LongTensor): index tensor of shape (L x kernel_size), having on each
                              row the indices of neighbors of each element of the input
                              a -1 indicates the absence of a neighbor, which is handled
                              as zero-padding
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`

    """
    def __init__(self, indices):
        super(IndexedAveragePool2d, self).__init__()
        self.logger = logging.getLogger(__name__ + '.IndexedAveragePool2d')
        self.indices = indices
        self.indices, self.mask = utils.prepare_mask(self.indices)

    def forward(self, input_images):
        self.logger.debug('Average pool image')

        self.indices = self.indices.to(input_images.device)
        self.mask = self.mask.to(input_images.device)

        col = input_images[..., self.indices] * self.mask

        out = torch.mean(col, 2)

        return out


class IndexedConv(nn.Module):
    r"""Applies a convolution over an input tensor where neighborhood relationships
    between elements are explicitly provided via an `indices` tensor.

    The output value of the layer with input size :math:`(N, C_{in}, L)` and output
    :math:`(N, C_{out}, L)` can be described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{c}=0}^{C_{in}-1}
                         \sum_{{i}=0}^{L-1}
                         \sum_{{k}=0}^{K} weight(C_{out_j}, c, k) * input(N_i, c, indices(i, k))
        \end{array}

    where

    | `indices` is a L x K tensor, where `K` is the size of the convolution kernel,
    | providing the indices of the `K` neighbors of input element `i`.
    | A -1 entry means zero-padding.

    Args:
        in_channels (int): Number of channels in the input tensor

        out_channels (int): Number of channels produced by the convolution

        indices (LongTensor): index tensor of shape (L x kernel_size), having on each
        row the indices of neighbors of each element of the input a -1 indicates the absence of a
        neighbor, which is handled as zero-padding

        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L)`
        - Output: :math:`(N, C_{out}, L)`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> indices = (10 * torch.rand(50, 3)).type(torch.LongTensor)
        >>> m = nn.IndexedConv(16, 3, indices)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)
    """

    def __init__(self, in_channels, out_channels, indices, bias=True):
        super(IndexedConv, self).__init__()
        self.logger = logging.getLogger(__name__ + '.IndexedConv')

        print('!!! bmm + no_grad version !!!')
        groups = 1

        kernel_size = indices.shape[0]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indices, self.mask = utils.prepare_mask(indices)
        self.output_width = self.indices.shape[1]

        self.register_buffer('indices_', self.indices)
        self.register_buffer('mask_', self.mask)

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        nbatch = input.shape[0]
        with torch.no_grad():
            col = input[..., self.indices_] * self.mask_
            # col is of shape (N, C_in, K, Wo)
            col = col.view(nbatch, -1, self.output_width)
        out = torch.bmm(self.weight.view(self.out_channels, -1).expand(nbatch, -1, -1), col)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)
        out = out.view(nbatch, self.out_channels, -1)

        return out

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
