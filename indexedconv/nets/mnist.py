"""
mnist.py
========
Contain the net for the mnist dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import indexedconv.utils as utils
from indexedconv.engine.torch import IndexedConv, IndexedMaxPool2d


class GLNet2HexaConvForMnist(nn.Module):
    r"""Network with indexed convolutions and pooling (square kernels).
    2 CL (after each conv layer, pooling is executed)
    1 FC

    Args:
        index_matrix (`torch.Tensor`): The index matrix corresponding to the input images.
    """

    def __init__(self, index_matrix):
        super(GLNet2HexaConvForMnist, self).__init__()

        index_matrix1 = index_matrix

        # Layer 1 : IndexedConv
        indices_conv1 = utils.neighbours_extraction(index_matrix1,
                                                    kernel_type='Hex')
        pool_indices1 = utils.neighbours_extraction(index_matrix1,
                                                    kernel_type='Hex', stride=2)
        self.cv1 = IndexedConv(1, 32, indices_conv1)
        self.max_pool1 = IndexedMaxPool2d(pool_indices1)

        # Layer 2 : IndexedConv
        index_matrix2 = utils.pool_index_matrix(index_matrix1, kernel_type='Hex', stride=2)
        indices_conv2 = utils.neighbours_extraction(index_matrix2,
                                                    kernel_type='Hex')
        pool_indices2 = utils.neighbours_extraction(index_matrix2,
                                                    kernel_type='Hex', stride=2)
        self.cv2 = IndexedConv(32, 64, indices_conv2)
        self.max_pool2 = IndexedMaxPool2d(pool_indices2)

        index_matrix3 = utils.pool_index_matrix(index_matrix2, kernel_type='Hex', stride=2)

        # Compute the number of pixels (where idx is not -1 in the index matrix) of the last features
        n_pixels = int(torch.sum(torch.ge(index_matrix3[0, 0], 0)).data)
        self.lin1 = nn.Linear(n_pixels * 64, 1024)

        self.lin2 = nn.Linear(1024, 10)

    def forward(self, x):
        drop = nn.Dropout(p=0.5)
        out = F.relu(self.cv1(x))
        out = self.max_pool1(out)
        out = F.relu(self.cv2(out))
        out = self.max_pool2(out)

        out = out.view(out.size(0), -1)

        out = self.lin1(out)
        out = drop(F.relu(out))

        out = self.lin2(out)

        return F.log_softmax(out, dim=1)
