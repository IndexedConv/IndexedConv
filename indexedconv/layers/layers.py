import torch
import torch.nn as nn
import torch.nn.functional as F
from ..engine import IndexedConv, IndexedAveragePool2d
from ..utils import neighbours_extraction, pool_index_matrix


class CReLU(nn.Module):
    """
    Concatenated ReLU as defiend in https://arxiv.org/abs/1603.05201

    """
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)


class SqueezeExcite(nn.Module):
    """
    Squeeze and excite the output of a convolution as described in the paper https://arxiv.org/abs/1709.01507
    Designed for vector data

    Args:
        num_channels (int): the number of channels of the feature map to squeeze and excite
        ratio (int): the ratio of the bottleneck

    """
    def __init__(self, num_channels, ratio):
        super(SqueezeExcite, self).__init__()
        reducted_channels = int(num_channels / ratio)
        self.reduction = nn.Linear(num_channels, reducted_channels)
        self.expand = nn.Linear(reducted_channels, num_channels)

    def forward(self, x):
        out = x.mean(dim=2)

        out = F.relu(self.reduction(out))
        out = F.sigmoid(self.expand(out))
        out = x * out.unsqueeze(2)

        return out


class SelfAttention(nn.Module):
    """
    Self attention layer as described in the SAGAN paper https://arxiv.org/abs/1805.08318
    Designed for vector data

    Args:
        channels (int): the number of channels of the feature map on which perform self-attention

    """
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.conv_f = nn.Conv1d(channels, channels // 8, kernel_size=1, bias=False)
        self.conv_g = nn.Conv1d(channels, channels // 8, kernel_size=1, bias=False)
        self.conv_h = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        batch = x.shape[0]
        channel = x.shape[1]
        f = self.conv_f(x.view(batch, channel, -1))
        g = self.conv_g(x.view(batch, channel, -1))
        h = self.conv_h(x.view(batch, channel, -1))

        s = torch.matmul(f.permute(0, 2, 1), g)

        beta = nn.functional.softmax(s, dim=-1)

        o = torch.matmul(beta, h.permute(0, 2, 1)).permute(0, 2, 1)

        return self.gamma * o.view(x.shape) + x


class DenseLayer(nn.Sequential):
    """
    Dense layer as defined in https://arxiv.org/abs/1608.06993
    With Indexed Convolution

    Args:
        num_input_features (int): the number of channels of the input
        growth_rate (int): the number of feature maps produced by the layer
        bn_size (int): the bottleneck size (in term of feature maps)
        drop_rate (float): the dropout rate. In PyTorch this is the probability for the neuron to be switched off
        bias (bool): whether or not using the bias of the convolutions
        indices (LongTensor): index tensor of shape (L x kernel_size),
            having on each row the indices of neighbors of each element of the input
            a -1 indicates the absence of a neighbor, which is handled as zero-padding
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, bias, indices):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=bias)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', IndexedConv(bn_size * growth_rate, growth_rate, indices, bias=bias)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    """
    Dense block as defined in https://arxiv.org/abs/1608.06993
    For details see :class:`~layers.DenseLayer`
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, bias, index_matrix):
        super(DenseBlock, self).__init__()
        indices = neighbours_extraction(index_matrix)
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, bias, indices)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    """
    Transition layer as defined in https://arxiv.org/abs/1608.06993
    Args:
        num_input_features (int): the number of channels of the input
        theta (float): the compression ratio
        bias (bool): whether or not using the bias of the convolutions
        index_matrix (torch.Tensor): the matrix of index for the images, shape(1, 1, matrix.size).
        camera_layout (str): the kernel shape, Hex for hexagonal Square for a square and Pool for a square of size 2.

    """
    def __init__(self, num_input_features, theta, bias, index_matrix, camera_layout):
        super(Transition, self).__init__()
        indices = neighbours_extraction(index_matrix, kernel_type=camera_layout, stride=2)
        self.pooled_matrix = pool_index_matrix(index_matrix, kernel_type=camera_layout, stride=2)
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, int(num_input_features * theta),
                                          kernel_size=1, stride=1, bias=bias))
        self.add_module('pool', IndexedAveragePool2d(indices))


class IndexedConvLayer(nn.Sequential):
    """
    Indexed convolution layer. A wrapper for IndexedConv - IndexedPooling - BatchNorm - Non Linearity

    Args:
        layer_id (int): the identifier of the layer
        index_matrix (torch.Tensor): the matrix of index for the images, shape(1, 1, matrix.size)
        num_input (int): the number of channels of the input
        num_output (int): the number of channels of the output
        non_linearity (nn.Module, optional): the non linearity. Default: ``nn.ReLU``
        pooling (nn.Module): the pooling method. Default: ``IndexedAveragePool2d``
        pooling_kernel (str): the shape of the pooling kernel. Default: ``Hex``
        pooling_radius (int): the radius of the pooling kernel. Default: ``1``
        pooling_stride (int): the stride of the pooling. Default: ``2``
        pooling_dilation (int): the dilation of the pooling kernel. Default: ``1``
        pooling_retina (bool): whether or using a retina like pooling kernel. Default: ``False``
        batchnorm (bool): whether or not using batchnorms. Default: ``True``
        drop_rate (int): the dropout rate. Default: ``0``
        bias (bool): whether or not using the bias of the convolutions. Default: ``True``
        kernel_type (str): the shape of the convolution kernel. Default: ``Hex``
        radius (int): the radius of the convolution kernel. Default: ``1``
        stride (int): the stride of the convolution. Default: ``1``
        dilation (int): the dilation of the convolution kernel. Default: ``1``
        retina (bool): whether or using a retina like convolution kernel. Default: ``False``
    """
    def __init__(self, layer_id, index_matrix, num_input, num_output, non_linearity=nn.ReLU,
                 pooling=IndexedAveragePool2d, pooling_kernel='Hex', pooling_radius=1, pooling_stride=2,
                 pooling_dilation=1, pooling_retina=False,
                 batchnorm=True, drop_rate=0, bias=True,
                 kernel_type='Hex', radius=1, stride=1, dilation=1, retina=False):
        super(IndexedConvLayer, self).__init__()
        self.drop_rate = drop_rate
        indices = neighbours_extraction(index_matrix, kernel_type, radius, stride, dilation, retina)
        self.index_matrix = pool_index_matrix(index_matrix, kernel_type=pooling_kernel, stride=1)
        self.add_module('cv'+layer_id, IndexedConv(num_input, num_output, indices, bias))
        if pooling is not None:
            p_indices = neighbours_extraction(self.index_matrix, pooling_kernel, pooling_radius, pooling_stride,
                                                    pooling_dilation, pooling_retina)
            self.index_matrix = pool_index_matrix(self.index_matrix, kernel_type=pooling_kernel, stride=pooling_stride)
            self.add_module('pool'+layer_id, pooling(p_indices))
        if batchnorm:
            self.add_module('bn'+layer_id, nn.BatchNorm1d(num_output))
        if non_linearity is not None:
            self.add_module('non_lin' + layer_id, non_linearity())

    def forward(self, x):
        new_features = super(IndexedConvLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class Regressor(nn.Module):
    """
    Multilayer perceptron for multitask regression.

    Args:
        tasks_name
        tasks_output
        num_features
        num_layers
        factor
        non_linearity=nn.ReLU
        batchnorm=True
        drop_rate=0
    """
    def __init__(self, tasks_name, tasks_output, num_features, num_layers, factor, non_linearity=nn.ReLU,
                 batchnorm=True, drop_rate=0):
        super(Regressor, self).__init__()
        for i, (task, output) in enumerate(zip(tasks_name, tasks_output)):
            t = nn.Sequential()
            for l in range(1, num_layers):
                if l == 1:
                    t.add_module('lin' + str(l) + '_' + task, nn.Linear(num_features, num_features // factor))
                else:
                    t.add_module('lin' + str(l) + '_' + task, nn.Linear(num_features // ((l - 1) * factor),
                                                                        num_features // (l * factor)))
                if batchnorm:
                    t.add_module('bn' + str(l) + '_' + task, nn.BatchNorm1d(num_features // (l * factor)))
                t.add_module('non_lin' + str(l) + '_' + task, non_linearity())

                if drop_rate > 0:
                    t.add_module('drop' + str(l) + '_' + task, nn.Dropout(p=drop_rate))
            t.add_module('output_' + task, nn.Linear(num_features // ((num_layers - 1) * factor), output))
            self.add_module(task, t)

    def forward(self, x):
        out = []
        for t in self.children():
            out.append(t(x))
        return torch.cat(out, dim=1)