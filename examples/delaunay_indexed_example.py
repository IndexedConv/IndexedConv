import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import indexedconv.utils as utils
from indexedconv.engine import IndexedConv, IndexedMaxPool2d

if __name__ == '__main__':

    channels = 1
    num_points = 25
    dist_max = 100
    positions = np.random.randn(num_points, 2) * 100
    data = torch.randn(1, channels, num_points)
    # Convolution
    indices = torch.tensor(utils.delaunay_vertices_neighbors_extraction(positions, dist_max))
    conv = IndexedConv(channels, 1, indices)

    # Pooling
    p_indices, pooled_positions = utils.delaunay_simplices_neighbors_extraction(positions, dist_max)
    pool = IndexedMaxPool2d(torch.tensor(p_indices))

    # Convolution 2
    indices_2 = torch.tensor(utils.delaunay_vertices_neighbors_extraction(pooled_positions, dist_max))
    conv2 = IndexedConv(1, 1, indices_2)

    # Pooling 2
    p_indices_2, pooled_positions_2 = utils.delaunay_simplices_neighbors_extraction(pooled_positions, dist_max)
    pool_2 = IndexedMaxPool2d(torch.tensor(p_indices_2))

    out_cv1 = conv(data)
    out_pool1 = pool(out_cv1)
    out_cv2 = conv2(out_pool1)
    out_pool2 = pool_2(out_cv2)

    fig, axs = plt.subplots(3, 2)
    xlim_right = positions[:, 0].max() + 10
    xlim_left = positions[:, 0].min() - 10
    ylim_right = positions[:, 1].max() + 10
    ylim_left = positions[:, 1].min() - 10

    axs[0, 0].scatter(positions[:, 0], positions[:, 1])
    for i in range(positions.shape[0]):
        axs[0, 0].annotate("{0:.2f}".format(data[0, 0, i].item()), (positions[i, 0], positions[i, 1]))
    # plt.triplot(tels['tel_x'], tels['tel_y'], simplices)
    axs[0, 0].set_xlim([xlim_left, xlim_right])
    axs[0, 0].set_ylim([ylim_left, ylim_right])
    axs[0, 0].set_title('Data, num points: {}'.format(num_points))

    axs[0, 1].scatter(positions[:, 0], positions[:, 1])
    for i in range(positions.shape[0]):
        axs[0, 1].annotate("{0:.2f}".format(out_cv1[0, 0, i].item()), (positions[i, 0], positions[i, 1]))
    axs[0, 1].set_xlim([xlim_left, xlim_right])
    axs[0, 1].set_ylim([ylim_left, ylim_right])
    axs[0, 1].set_title('First convolution')

    axs[1, 0].scatter(pooled_positions[:, 0], pooled_positions[:, 1])
    for i in range(pooled_positions.shape[0]):
        axs[1, 0].annotate("{0:.2f}".format(out_pool1[0, 0, i].item()), (pooled_positions[i, 0], pooled_positions[i, 1]))
    axs[1, 0].set_xlim([xlim_left, xlim_right])
    axs[1, 0].set_ylim([ylim_left, ylim_right])
    axs[1, 0].set_title('First pooling, num points: {}'.format(pooled_positions.shape[0]))

    axs[1, 1].scatter(pooled_positions[:, 0], pooled_positions[:, 1])
    for i in range(pooled_positions.shape[0]):
        axs[1, 1].annotate("{0:.2f}".format(out_cv2[0, 0, i].item()), (pooled_positions[i, 0], pooled_positions[i, 1]))
    axs[1, 1].set_xlim([xlim_left, xlim_right])
    axs[1, 1].set_ylim([ylim_left, ylim_right])
    axs[1, 1].set_title('Second convolution')

    axs[2, 0].scatter(pooled_positions_2[:, 0], pooled_positions_2[:, 1])
    for i in range(pooled_positions_2.shape[0]):
        axs[2, 0].annotate("{0:.2f}".format(out_pool2[0, 0, i].item()),
                           (pooled_positions_2[i, 0], pooled_positions_2[i, 1]))
    axs[2, 0].set_xlim([xlim_left, xlim_right])
    axs[2, 0].set_ylim([ylim_left, ylim_right])
    axs[2, 0].set_title('Second pooling, num points: {}'.format(pooled_positions_2.shape[0]))

    plt.axis('equal')
    plt.show()

