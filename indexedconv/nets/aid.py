"""
aid.py
========
Contain nets for the aid dataset
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from indexedconv.engine import MaskedConv2d
import indexedconv.utils as utils
import indexedconv.engine as engine


class WideNetIndexConvIndexPool(nn.Module):
    """ResNet like Network from HexaConv paper implemented with indexed convolutions and pooling.

    Args:
        index_matrix (`torch.Tensor`): The index matrix corresponding to the input images.
        camera_layout (str): The grid shape of the images.
        n_out (int): Number of features after last convolution.
    """
    def __init__(self, index_matrix, camera_layout, n_out):
        super(WideNetIndexConvIndexPool, self).__init__()

        index_matrix0 = index_matrix

        try:
            camera_layout in ['Hex', 'Square']
        except ValueError:
            print('Unkown camera layout {}'.format(camera_layout))

        if camera_layout == 'Hex':
            n1 = 42
            n2 = 83
            n3 = 166
        elif camera_layout == 'Square':
            n1 = 37
            n2 = 74
            n3 = 146

        # Layer 0
        indices_conv0 = utils.neighbours_extraction(index_matrix0,
                                                    kernel_type=camera_layout,
                                                    stride=2)
        self.cv0 = engine.IndexedConv(3, n1, indices_conv0)

        # Layer 1 : IndexedConv
        index_matrix1 = utils.pool_index_matrix(index_matrix0)
        indices_conv1 = utils.neighbours_extraction(index_matrix1,
                                                    kernel_type=camera_layout)
        self.cv1_1 = engine.IndexedConv(n1, n1, indices_conv1)
        self.cv1_2 = engine.IndexedConv(n1, n1, indices_conv1)
        self.cv1_3 = engine.IndexedConv(n1, n1, indices_conv1)
        self.cv1_4 = engine.IndexedConv(n1, n1, indices_conv1)

        self.bn1_1 = nn.BatchNorm1d(n1)
        self.bn1_2 = nn.BatchNorm1d(n1)
        self.bn1_3 = nn.BatchNorm1d(n1)
        self.bn1_4 = nn.BatchNorm1d(n1)

        indices_res_conv1 = utils.neighbours_extraction(index_matrix1,
                                                        kernel_type='One', stride=2)
        self.res_cv1to2 = engine.IndexedConv(n1, n2, indices_res_conv1)

        # Layer 2 : IndexedConv
        indices_conv1to2 = utils.neighbours_extraction(index_matrix1,
                                                    kernel_type=camera_layout, stride=2)
        self.cv2_1 = engine.IndexedConv(n1, n2, indices_conv1to2)

        index_matrix2 = utils.pool_index_matrix(index_matrix1, kernel_type=camera_layout, stride=2)
        indices_conv2 = utils.neighbours_extraction(index_matrix2,
                                                    kernel_type=camera_layout)
        self.cv2_2 = engine.IndexedConv(n2, n2, indices_conv2)
        self.cv2_3 = engine.IndexedConv(n2, n2, indices_conv2)
        self.cv2_4 = engine.IndexedConv(n2, n2, indices_conv2)

        self.bn2_1 = nn.BatchNorm1d(n1)
        self.bn2_2 = nn.BatchNorm1d(n2)
        self.bn2_3 = nn.BatchNorm1d(n2)
        self.bn2_4 = nn.BatchNorm1d(n2)

        indices_res_conv2 = utils.neighbours_extraction(index_matrix2,
                                                        kernel_type='One', stride=2)
        self.res_cv2to3 = engine.IndexedConv(n2, n3, indices_res_conv2)

        # Layer 3 : IndexedConv
        indices_conv2to3 = utils.neighbours_extraction(index_matrix2,
                                                       kernel_type=camera_layout, stride=2)
        self.cv3_1 = engine.IndexedConv(n2, n3, indices_conv2to3)
        index_matrix3 = utils.pool_index_matrix(index_matrix2, kernel_type=camera_layout, stride=2)
        indices_conv3 = utils.neighbours_extraction(index_matrix3,
                                                    kernel_type=camera_layout)
        self.cv3_2 = engine.IndexedConv(n3, n3, indices_conv3)
        self.cv3_3 = engine.IndexedConv(n3, n3, indices_conv3)
        self.cv3_4 = engine.IndexedConv(n3, n3, indices_conv3)

        self.bn3_1 = nn.BatchNorm1d(n2)
        self.bn3_2 = nn.BatchNorm1d(n3)
        self.bn3_3 = nn.BatchNorm1d(n3)
        self.bn3_4 = nn.BatchNorm1d(n3)

        self.bn4 = nn.BatchNorm1d(n3)

        self.conv4 = nn.Conv1d(in_channels=n3, out_channels=n_out, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, engine.IndexedConv):
                if m.kernel_size == 7:
                    ks = 9
                else:
                    ks = m.kernel_size
                m.weight.data.normal_(0.0, np.sqrt(2 / (ks * m.in_channels)))
            if isinstance(m, nn.Conv1d):
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                m.weight.data.normal_(0.0, np.sqrt(2 / n))

    def forward(self, x):
        dropout = nn.Dropout(p=0)

        out = self.cv0(x)
        res1 = out
        out = dropout(self.cv1_1(F.relu(self.bn1_1(out))))
        out = self.cv1_2(F.relu(self.bn1_2(out)))
        res2 = out + res1
        out = dropout(self.cv1_3(F.relu(self.bn1_3(res2))))
        out = self.cv1_4(F.relu(self.bn1_4(out)))
        res3 = out + res2
        res4_2 = self.res_cv1to2(res3)  # Residual link down sampled

        out = dropout(self.cv2_1(F.relu(self.bn2_1(res3))))
        out = self.cv2_2(F.relu(self.bn2_2(out)))
        res5 = out + res4_2
        out = dropout(self.cv2_3(F.relu(self.bn2_3(res5))))
        out = self.cv2_4(F.relu(self.bn2_4(out)))
        res6 = out + res5
        res8_2 = self.res_cv2to3(res6)  # Residual link down sampled

        out = dropout(self.cv3_1(F.relu(self.bn3_1(res6))))
        out = self.cv3_2(F.relu(self.bn3_2(out)))
        res9 = out + res8_2
        out = dropout(self.cv3_3(F.relu(self.bn3_3(res9))))
        out = self.cv3_4(F.relu(self.bn3_4(out)))
        res10 = out + res9

        out = F.relu(self.bn4(res10))

        out = F.avg_pool1d(out, kernel_size=out.shape[2:], stride=1, padding=0)

        out = self.conv4(out)

        out = out.view(out.size(0), -1)

        return F.log_softmax(out, dim=1)


class WideNet(nn.Module):
    """ResNet like Network from HexaConv paper (Z²).

    Args:
        n_out (int): Number of features after last convolution.
    """

    def __init__(self, n_out):
        super(WideNet, self).__init__()

        n1 = 37
        n2 = 74
        n3 = 146

        self.cv0 = nn.Conv2d(3, n1, kernel_size=3, stride=2, padding=1)

        # Layer 1 :
        self.cv1_1 = nn.Conv2d(n1, n1, kernel_size=3, stride=1, padding=1)
        self.cv1_2 = nn.Conv2d(n1, n1, kernel_size=3, stride=1, padding=1)
        self.cv1_3 = nn.Conv2d(n1, n1, kernel_size=3, stride=1, padding=1)
        self.cv1_4 = nn.Conv2d(n1, n1, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(n1)
        self.bn1_2 = nn.BatchNorm2d(n1)
        self.bn1_3 = nn.BatchNorm2d(n1)
        self.bn1_4 = nn.BatchNorm2d(n1)

        self.res_cv1to2 = nn.Conv2d(n1, n2, kernel_size=1, stride=2)

        # Layer 2
        self.cv2_1 = nn.Conv2d(n1, n2, kernel_size=3, stride=2, padding=1)

        self.cv2_2 = nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1)
        self.cv2_3 = nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1)
        self.cv2_4 = nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1)

        self.bn2_1 = nn.BatchNorm2d(n1)
        self.bn2_2 = nn.BatchNorm2d(n2)
        self.bn2_3 = nn.BatchNorm2d(n2)
        self.bn2_4 = nn.BatchNorm2d(n2)

        self.res_cv2to3 = nn.Conv2d(n2, n3, kernel_size=1, stride=2)

        # Layer 3
        self.cv3_1 = nn.Conv2d(n2, n3, kernel_size=3, stride=2, padding=1)

        self.cv3_2 = nn.Conv2d(n3, n3, kernel_size=3, stride=1, padding=1)
        self.cv3_3 = nn.Conv2d(n3, n3, kernel_size=3, stride=1, padding=1)
        self.cv3_4 = nn.Conv2d(n3, n3, kernel_size=3, stride=1, padding=1)

        self.bn3_1 = nn.BatchNorm2d(n2)
        self.bn3_2 = nn.BatchNorm2d(n3)
        self.bn3_3 = nn.BatchNorm2d(n3)
        self.bn3_4 = nn.BatchNorm2d(n3)

        self.bn4 = nn.BatchNorm2d(n3)

        self.conv4 = nn.Conv1d(in_channels=n3, out_channels=n_out, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) | isinstance(m, nn.Conv1d):
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                m.weight.data.normal_(0.0, np.sqrt(2 / n))

    def forward(self, x):
        dropout = nn.Dropout(p=0)
        out = self.cv0(x)
        res1 = out
        out = dropout(self.cv1_1(F.relu(self.bn1_1(out))))
        out = self.cv1_2(F.relu(self.bn1_2(out)))
        res2 = out + res1
        out = dropout(self.cv1_3(F.relu(self.bn1_3(res2))))
        out = self.cv1_4(F.relu(self.bn1_4(out)))
        res3 = out + res2
        res4_2 = self.res_cv1to2(res3)  # Residual link down sampled

        out = dropout(self.cv2_1(F.relu(self.bn2_1(res3))))
        out = self.cv2_2(F.relu(self.bn2_2(out)))
        res5 = out + res4_2
        out = dropout(self.cv2_3(F.relu(self.bn2_3(res5))))
        out = self.cv2_4(F.relu(self.bn2_4(out)))
        res6 = out + res5
        res8_2 = self.res_cv2to3(res6)  # Residual link down sampled

        out = dropout(self.cv3_1(F.relu(self.bn3_1(res6))))
        out = self.cv3_2(F.relu(self.bn3_2(out)))
        res9 = out + res8_2
        out = dropout(self.cv3_3(F.relu(self.bn3_3(res9))))
        out = self.cv3_4(F.relu(self.bn3_4(out)))
        res10 = out + res9

        out = F.relu(self.bn4(res10))

        out = F.avg_pool2d(out, kernel_size=out.shape[2:], stride=1, padding=0)

        out = self.conv4(out.view(out.shape[0], out.shape[1], -1))

        out = out.view(out.size(0), -1)

        return F.log_softmax(out, dim=1)


class WideNetMasked(nn.Module):
    """ResNet like Network from HexaConv paper implementing masked convolutions.

    Args:
        n_out (int): Number of features after last convolution.
    """

    def __init__(self, n_out):
        super(WideNetMasked, self).__init__()

        n1 = 42
        n2 = 83
        n3 = 166

        self.cv0 = MaskedConv2d(3, n1, kernel_size=3, stride=2, padding=1)

        # Layer 1 :
        self.cv1_1 = MaskedConv2d(n1, n1, kernel_size=3, stride=1, padding=1)
        self.cv1_2 = MaskedConv2d(n1, n1, kernel_size=3, stride=1, padding=1)
        self.cv1_3 = MaskedConv2d(n1, n1, kernel_size=3, stride=1, padding=1)
        self.cv1_4 = MaskedConv2d(n1, n1, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(n1)
        self.bn1_2 = nn.BatchNorm2d(n1)
        self.bn1_3 = nn.BatchNorm2d(n1)
        self.bn1_4 = nn.BatchNorm2d(n1)

        self.res_cv1to2 = nn.Conv2d(n1, n2, kernel_size=1, stride=2, padding=0)

        # Layer 2
        self.cv2_1 = MaskedConv2d(n1, n2, kernel_size=3, stride=2, padding=1)

        self.cv2_2 = MaskedConv2d(n2, n2, kernel_size=3, stride=1, padding=1)
        self.cv2_3 = MaskedConv2d(n2, n2, kernel_size=3, stride=1, padding=1)
        self.cv2_4 = MaskedConv2d(n2, n2, kernel_size=3, stride=1, padding=1)

        self.bn2_1 = nn.BatchNorm2d(n1)
        self.bn2_2 = nn.BatchNorm2d(n2)
        self.bn2_3 = nn.BatchNorm2d(n2)
        self.bn2_4 = nn.BatchNorm2d(n2)

        self.res_cv2to3 = nn.Conv2d(n2, n3, kernel_size=1, stride=2, padding=0)

        # Layer 3
        self.cv3_1 = MaskedConv2d(n2, n3, kernel_size=3, stride=2, padding=1)

        self.cv3_2 = MaskedConv2d(n3, n3, kernel_size=3, stride=1, padding=1)
        self.cv3_3 = MaskedConv2d(n3, n3, kernel_size=3, stride=1, padding=1)
        self.cv3_4 = MaskedConv2d(n3, n3, kernel_size=3, stride=1, padding=1)

        self.bn3_1 = nn.BatchNorm2d(n2)
        self.bn3_2 = nn.BatchNorm2d(n3)
        self.bn3_3 = nn.BatchNorm2d(n3)
        self.bn3_4 = nn.BatchNorm2d(n3)

        self.bn4 = nn.BatchNorm2d(n3)

        self.conv4 = nn.Conv1d(in_channels=n3, out_channels=n_out, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, MaskedConv2d) | isinstance(m, nn.Conv1d) | isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                m.weight.data.normal_(0.0, np.sqrt(2 / n))

    def forward(self, x):
        dropout = nn.Dropout(p=0)
        out = self.cv0(x)
        res1 = out
        out = dropout(self.cv1_1(F.relu(self.bn1_1(out))))
        out = self.cv1_2(F.relu(self.bn1_2(out)))
        res2 = out + res1
        out = dropout(self.cv1_3(F.relu(self.bn1_3(res2))))
        out = self.cv1_4(F.relu(self.bn1_4(out)))
        res3 = out + res2
        res4_2 = self.res_cv1to2(res3)  # Residual link down sampled

        out = dropout(self.cv2_1(F.relu(self.bn2_1(res3))))
        out = self.cv2_2(F.relu(self.bn2_2(out)))
        res5 = out + res4_2
        out = dropout(self.cv2_3(F.relu(self.bn2_3(res5))))
        out = self.cv2_4(F.relu(self.bn2_4(out)))
        res6 = out + res5
        res8_2 = self.res_cv2to3(res6)  # Residual link down sampled

        out = dropout(self.cv3_1(F.relu(self.bn3_1(res6))))
        out = self.cv3_2(F.relu(self.bn3_2(out)))
        res9 = out + res8_2
        out = dropout(self.cv3_3(F.relu(self.bn3_3(res9))))
        out = self.cv3_4(F.relu(self.bn3_4(out)))
        res10 = out + res9

        out = F.relu(self.bn4(res10))

        out = F.avg_pool2d(out, kernel_size=out.shape[2:], stride=1, padding=0)

        out = self.conv4(out.view(out.shape[0], out.shape[1], -1))

        out = out.view(out.size(0), -1)

        return F.log_softmax(out, dim=1)
