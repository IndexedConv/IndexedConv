"""
utils.py
========
Contain utility functions for the indexed convolution
"""

import logging
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
from scipy.spatial import Delaunay


def get_gpu_usage_map(device_id):
    """Get the current gpu usage.
    Inspired from `gpu usage from
    pytorch <https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4>`_

    Args:
        device_id (int): the GPU id as GPU/Unit's 0-based index in the natural enumeration returned  by the driver

    Returns
        dict - usage
            Keys are device ids as integers.
            Values are memory usage as integers in MB, total memory usage as integers in MB, gpu utilization in %.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--id=' + str(device_id), '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    # gpu_usage = [[int(y) for y in x.split(',')] for x in result.strip().split('\n')]
    gpu_usage = [int(y) for y in result.split(',')]
    # gpu_usage_map = {}
    gpu_usage_map = {'memory_used': gpu_usage[0], 'total_memory': gpu_usage[1], 'utilization': gpu_usage[2]}
    # for i, gpu in enumerate(gpu_usage):
    #     gpu_usage_map[i] = {'memory_used': gpu[0], 'total_memory': gpu[1], 'utilization': gpu[2]}
    return gpu_usage_map


def compute_total_parameter_number(net):
    r"""Computes the total number of parameters of a network.

    Args:
        net (:class:`nn.Module`): The network.

    Returns:
        A int
    """
    num_parameters = 0
    for name, param in net.named_parameters():
        num_parameters += param.clone().cpu().data.view(-1).size(0)

    return num_parameters


#####################
# Indexed functions #
#####################

def create_index_matrix(nbRow, nbCol, injTable):
    r"""Creates the matrix of index of the pixels of the images of any shape stored as vectors.

    Args:
        nbRow (int): The number of rows of the index matrix.
        nbCol (int): The number of cols of the index matrix.
        injTable (numpy.array): The injunction table, i.e. the list of the position of every pixels of the vector image
            in a vectorized square image.

    Returns:
        A torch.Tensor containing the index of each pixel represented in a matrix.

    Example:
        >>> image = [0, 1, 2, 3, 4, 5, 6]  # hexagonal image stored as a vector
        >>> # in the hexagonal space                  0 1
        >>> #                                        2 3 4
        >>> #                                         5 6
        >>> # injunction table of the pixel position of a hexagonal image represented in the axial addressing system
        >>> injTable = [0, 1, 3, 4, 5, 7, 8]
        >>> index_matrix = [[0, 1, -1], [2, 3, 4], [-1, 5, 6]]
        [[0, 1, -1],
        [2,  3, 4],
        [-1, 5, 6]]
    """
    logger = logging.getLogger(__name__ + '.create_index_matrix')
    index_matrix = torch.full((int(nbRow), int(nbCol)), -1)
    for i, idx in enumerate(injTable):
        idx_row = int(idx // nbRow)
        idx_col = int(idx % nbCol)
        index_matrix[idx_row,idx_col] = i

    index_matrix.unsqueeze_(0)
    index_matrix.unsqueeze_(0)
    return index_matrix


def img2mat(input_images, index_matrix):
    """Transforms a batch of features of vector images in a batch of features of matrix images.

     Args:
         input_images (torch.Tensor): The images with shape (batch, features, image).
         index_matrix (torch.Tensor): The index matrix containing the index of the pixels of the images.
            represented in a matrix

     Returns:
         A torch.Tensor
     """
    logger = logging.getLogger(__name__ + '.img2mat')
    # First create a tensor of shape : batch, features, index_matrix.size filled with zeros
    image_matrix = input_images.new_zeros((input_images.size(0),
                                           input_images.size(1),
                                           index_matrix.size(-2),
                                           index_matrix.size(-1)), dtype=torch.int)

    logger.debug('image matrix shape : {}'.format(image_matrix.size()))

    for i in range(index_matrix.size(-2)):  # iterate over the rows of index matrix
        for j in range(index_matrix.size(-1)):  # iterate over the cols of index matrix
            if index_matrix.data[0, 0, i, j] != -1:
                image_matrix[:, :, i, j] = input_images[:, :, int(index_matrix[0, 0, i, j])]

    return image_matrix


def mat2img(input_matrix, index_matrix):
    """
    Transforms a batch of features of matrix images in a batch of features of vector images.

    Args:
        input_matrix (torch.Tensor): The images with shape (batch, features, matrix.size).
        index_matrix (torch.Tensor): The index matrix for the images, shape(1, 1, matrix.size).

    """
    logger = logging.getLogger(__name__ + '.mat2img')
    logger.debug('input matrix shape : {}'.format(input_matrix.size()))
    image_length = index_matrix[0, 0, torch.ge(index_matrix[0, 0], 0)].size(0)

    logger.debug('new image length : {}'.format(image_length))

    images = input_matrix.new_zeros((input_matrix.size(0), input_matrix.size(1), image_length), dtype=torch.float)

    logger.debug('new images shape : {}'.format(images.size()))

    for i in range(index_matrix.size(-2)):  # iterate over the rows of index matrix
        for j in range(index_matrix.size(-1)):  # iterate over the cols of index matrix
            if index_matrix[0, 0, i, j] != -1:
                images[:, :, int(index_matrix[0, 0, i, j])] = input_matrix[:, :,  i, j]

    return images


def pool_index_matrix(index_matrix, kernel_type='Pool', stride=2):
    """
    Pools an index matrix.

    Args:
        index_matrix (torch.Tensor): The index matrix for the images, shape(1, 1, matrix.size).
        kernel_type (str): The kernel shape, Hex for hexagonal, Square for a square of size 3
            and Pool for a square of size 2.
        stride (int): The stride.

    Returns:
        A torch.Tensor containing the pooled matrix.
    """
    logger = logging.getLogger(__name__ + '.pool_index_matrix')
    if kernel_type == 'Pool':
        weight = torch.Tensor([[1, 0], [0, 0]]).requires_grad_(False)
        padding = 0
    elif kernel_type == 'Square' or kernel_type == 'Hex':
        weight = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).requires_grad_(False)
        padding = 1
    weight.unsqueeze_(0)
    weight.unsqueeze_(0)

    pooled_matrix = F.conv2d(index_matrix, weight, stride=stride, padding=padding).data

    logger.debug('pooled matrix shape : {}'.format(pooled_matrix.size()))
    # Index reconstruction
    idx = 0
    for i in range(pooled_matrix.size(-2)):  # iterate over the rows of index matrix
        for j in range(pooled_matrix.size(-1)):  # iterate over the cols of index matrix
            if pooled_matrix[0, 0, i, j] != -1:
                pooled_matrix[0, 0, i, j] = idx
                idx += 1

    return pooled_matrix


def build_kernel(kernel_type, radius=1, dilation=1):
    """Builds the convolution kernel or mask. (Following the suggestion of Miguel Lallena).

    Args:
        kernel_type (str): The type of kernel. Can be hexagonal ('Hex') or square ('Square').
        radius (int): The radius of the kernel.
        dilation (int): The dilation. A dilation of 1 means no dilation.

    Returns:
         A :class:`np.array`.
    """
    k_size = 2 * radius * dilation + 1
    kernel = np.zeros((k_size, k_size))
    for i in range(0, k_size, dilation):
        for j in range(0, k_size, dilation):
            if kernel_type == 'Square':
                kernel[i, j] = 1
            elif kernel_type == 'Hex':
                kernel[i, j] = int(np.abs(i -j) <= radius * dilation)
            else:
                raise ValueError('Unknown kernel type')

    return kernel


def neighbours_extraction(index_matrix, kernel_type='Hex', radius=1, stride=1, dilation=1, retina=False):
    """Builds the matrix of indices from an index matrix based on a kernel.

    The matrix of indices contains for each pixel of interest its neighbours, including itself.

    Args:
        index_matrix (torch.Tensor): Matrix of index for the images, shape(1, 1, matrix.size).
        kernel_type (str): The kernel shape, Hex for hexagonal Square for a square and Pool for a square of size 2.
        radius (int): The radius of the kernel.
        stride (int): The stride.
        dilation (int): The dilation. A dilation of 1 means no dilation.
        retina (bool): Whether to build a retina like kernel. If True, dilation must be 1.

    Returns:
        A torch.Tensor - the matrix of the neighbours.

    Example:
        >>> index_matrix = [[0, 1, -1], [2, 3, 4], [-1, 5, 6]]
        [[0, 1, -1],
        [2,  3, 4],
        [-1, 5, 6]]
        >>> kernel_type = 'Hex'
        >>> radius = 1
        >>> kernel
        [[1, 1, 0],
        [ 1, 1, 1],
        [ 0, 1, 1]]
        >>> stride = 1
        >>> neighbours = neighbours_extraction(index_matrix, kernel_type, radius, stride)
        [[-1, -1, -1,  0,  1,  2,  3],
        [ -1, -1,  0,  1, -1,  3,  4],
        [ -1,  0, -1,  2,  3, -1,  5],
        [  0,  1,  2,  3,  4,  5,  6],
        [  1, -1,  3,  4, -1,  6, -1],
        [  2,  3, -1,  5,  6, -1, -1],
        [  3,  4,  5,  6, -1, -1, -1]]
    """
    if retina:
        dilation = 1
    padding = radius * dilation * 2
    stride = stride
    bound = radius * dilation * 2 if radius > 0 else 1
    if kernel_type == 'Pool':
        kernel = np.ones((2, 2), dtype=bool)
        stride = 2
        bound = 1
        padding = 0
        center = 0
    elif retina:
        kernel = build_kernel(kernel_type, 1, radius).astype(bool)
        for i in range(1, radius):
            sub_kernel = np.zeros_like(kernel).astype(bool)
            sub_kernel[i:sub_kernel.shape[0]-i, i:sub_kernel.shape[1]-i] = build_kernel(kernel_type, 1, radius - i).astype(bool)
            kernel = kernel + sub_kernel
        center = int((np.count_nonzero(kernel) - 1) / 2)
    else:
        kernel = build_kernel(kernel_type, radius, dilation).astype(bool)
        center = int((np.count_nonzero(kernel) - 1) / 2)

    neighbours = []

    idx_mtx = np.ones((index_matrix.size(-2)+padding, index_matrix.size(-1)+padding), dtype=int) * (-1)
    offset = int(padding/2)
    if offset == 0:
        idx_mtx = index_matrix[0, 0, :, :].numpy()
    else:
        idx_mtx[offset:-offset, offset:-offset] = index_matrix[0, 0, :, :].numpy()

    for i in range(0, idx_mtx.shape[0]-bound, stride):
        for j in range(0, idx_mtx.shape[1]-bound, stride):
            patch = idx_mtx[i:i+kernel.shape[0], j:j+kernel.shape[1]][kernel]
            if patch[center] == -1:
                continue
            neighbours.append(patch)

    neighbours = np.asarray(neighbours).T
    neighbours = torch.from_numpy(neighbours).long()

    return neighbours


def clean_sims(points, sims, neighbors, thresh=350):

    keep_sims = np.ones(sims.shape[0], dtype=bool)
    for i, sim in enumerate(sims):
        for l in range(3):
            edge_idx0 = sim[l]
            edge_idx1 = sim[(l + 1) % 3]
            p0 = points[edge_idx0]
            p1 = points[edge_idx1]
            if np.linalg.norm(p1 - p0) > thresh:
                keep_sims[i] = False
                break
    sims_map = {}
    j = 0
    for i in range(sims.shape[0]):
        if keep_sims[i]:
            sims_map[i] = j
            j += 1
    for neighb in neighbors:
        for i, n in enumerate(neighb):
            if n in sims_map.keys():
                neighb[i] = sims_map[n]
            else:
                neighb[i] = -1
    sims = sims[keep_sims]
    neighbors = neighbors[keep_sims]

    return np.array(sims), np.array(neighbors)


def delaunay_simplices_neighbors_extraction(positions, max_distance):

    tri = Delaunay(positions)
    simplices, neighbors = clean_sims(tri.points, tri.simplices, tri.neighbors, thresh=max_distance)
    used_sims = np.zeros(simplices.shape[0], dtype=bool)
    neighborhood = []
    barycentres = []
    for i, sim in enumerate(simplices):
        if not used_sims[i]:
            found_neighbor = False
            for sim_neighbor in neighbors[i]:
                if not used_sims[sim_neighbor] and not sim_neighbor == -1:
                    neighb = np.concatenate([sim, simplices[sim_neighbor]])
                    used_sims[sim_neighbor] = True
                    found_neighbor = True
                    break
            if not found_neighbor:
                neighb = sim
            used_sims[i] = True
            neighb = list(set(list(neighb)))
#             neighb.sort()
            neighborhood.append(neighb)
            pos = positions[neighb]
            length = pos.shape[0]
            sum_x = np.sum(pos[:, 0])
            sum_y = np.sum(pos[:, 1])
            barycentres.append([sum_x / length, sum_y / length])
        np_neighborhood = np.full([len(neighborhood), len(max(neighborhood, key=lambda x: len(x)))], -1)
        for i, j in enumerate(neighborhood):
            np_neighborhood[i][0:len(j)] = j

        return np_neighborhood, np.array(barycentres)


def clean_vertices(points, neighbors, thresh=350):
    for i, neighb in enumerate(neighbors):
        for j, n in enumerate(neighb):
            p0 = points[i]
            p1 = points[n]
            if np.linalg.norm(p1 - p0) >  thresh:
                neighb[j] = -1
    return neighbors


def delaunay_vertices_neighbors_extraction(positions, max_distance):
    tri = Delaunay(positions)
    idptr, vertices = tri.vertex_neighbor_vertices
    neighbors = []
    for i in range(idptr.shape[0]-1):
        start_ptr = idptr[i]
        end_ptr = idptr[i+1]
        neighbors.append(list(vertices[start_ptr:end_ptr]))
    neighbors = clean_vertices(tri.points, neighbors, max_distance)
    np_neighbors = np.full([len(neighbors),len(max(neighbors,key = lambda x: len(x)))], -1)
    for i,j in enumerate(neighbors):
        np_neighbors[i][0:len(j)] = j
    return np_neighbors


def prepare_mask(indices):
    """Prepares the indices and the mask for the GEMM im2col operation.

    Args:
        indices (torch.Tensor): The matrix of indices containing the neighbours of each pixel of interest.

    """
    padded = indices == -1
    new_indices = indices.clone()
    new_indices[padded] = 0

    mask = torch.FloatTensor([1, 0])
    mask = mask[..., padded.long()]

    return new_indices, mask


###################
# Transformations #
###################

def square_to_hexagonal_index_matrix(image):
    """Creates the index matrix of square images in a hexagonal grid (axial addressing system).

    Args:
        image: input tensor of shape (c, n, m)

        """
    index_matrix = torch.ones(image.shape[1],
                              image.shape[2] + int(np.ceil(image.shape[1] / 2))) * -1
    n = 0
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            index_matrix[i, j + int(np.ceil(i / 2))] = n
            n += 1
    return index_matrix


def square_to_hexagonal(image):
    """Rough sampling of square images to hexagonal grid

    Args:
        image: image tensor of shape (c, n, m)

    """
    image_tr = image.clone()
    image_tr[:, :, :-1] = image[:, :, 1:]
    image_mean = (image + image_tr) / 2
    image[:, 1::2, :] = image_mean[:, 1::2, :]

    return image.view(image.shape[0], -1)


def build_hexagonal_position(index_matrix):
    """Computes the position of the pixels in the hexagonal grid from the index matrix.

    Args:
        index_matrix (tensor): The index matrix representing the index of each pixel in the axial addressing system.
    """
    pix_positions = []
    for i in range(index_matrix.shape[0]):
        for j in range(index_matrix.shape[1]):
            if not index_matrix[i, j] == -1:
                pix_positions.append([j - i/2, -i])
    return pix_positions


def normalize(images):
    """Normalizes images.

    Args:
        images: image tensor of shape (c, n, m)

    """
    images -= np.mean(images, axis=(1, 2), keepdims=True)
    std = np.sqrt(images.var(axis=(1, 2), ddof=1, keepdims=True))
    std[std < 1e-8] = 1.
    images /= std
    return images


class PCA(object):
    """ Credit E. Hoogeboom http://github.com/ehoogeboom/hexaconv"""
    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        """
        The computation works as follows:

        - The covariance is :code:`C = 1/(n-1) * D * D.T`

        - The eigendecomp of C is: :code:`C = V Sigma V.T`

        - Let :code:`Y = 1/sqrt(n-1) * D`

        - Let :code:`U S V = svd(Y)`,

        - Then the columns of U are the eigenvectors of :code:`Y * Y.T = C`

        - And the singular values S are the sqrts of the eigenvalues of C

        - We can apply PCA by multiplying by U.T
        """

        # We require scaled, zero-mean data to SVD,
        # But we don't want to copy or modify user data
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=False, ZCA=False, regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by :math:`\sigma^{-1/2} U.T`

        Any orthogonal transformation of this is also white,
        and when :code:`ZCA=True` we choose :math:`U \sigma^{-1/2} U.T`
        """
        if whiten:
            # Compute :math:`\sigma^{-1/2} = S^{-1}`,
            # with smoothing for numerical stability
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                # The ZCA whitening matrix
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                # The whitening matrix
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)


###################
# Data            #
###################

class NumpyDataset(Dataset):
    """Loads data in a Dataset from :class:`numpy.array`.

    Args:
        data (numpy.array): The data to load.
        labels (numpy.array): The labels of the data.
        transform (callable, optional): A callable or a composition of callable to be applied to the data.
        target_transform (callable, optional): A callable or a composition of callable to be applied to the labels.
    """
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.images = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        if self.transform:
            self.images[item] = self.transform(self.images[item])
        if self.target_transform:
            self.labels[item] = self.target_transform(self.labels[item])

        return self.images[item], self.labels[item]


class HDF5Dataset(Dataset):
    """Loads data in a Dataset from a HDF5 file.

    Args:
        path (str): The path to the HDF5 file.
        transform (callable, optional): A callable or a composition of callable to be applied to the data.
        target_transform (callable, optional): A callable or a composition of callable to be applied to the labels.
    """
    def __init__(self, path, transform=None, target_transform=None):
        with h5py.File(path, 'r') as f:
            self.images = f['images'][()]
            self.labels = f['labels'][()]
        self.transform = transform
        self.target_transform = target_transform
        self.logger = logging.getLogger(__name__ + 'HDF5Dataset')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.transform(label)

        return image, label


# Transforms
class NumpyToTensor(object):
    """Converts a numpy array to a tensor."""

    def __call__(self, data):

        data = torch.tensor(data)
        return data


class SquareToHexa(object):
    """Converts an image with a square grid to an image with a hexagonal one."""
    def __call__(self, image):
        image = square_to_hexagonal(image)
        # print(sample)
        return image
