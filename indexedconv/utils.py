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


def get_gpu_usage_map(device_id):
    """Get the current gpu usage.
    inspired from : https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    Parameters
    ----------
    device_id (int): the GPU id as GPU/Unit's 0-based index in the natural enumeration returned  by
       the driver
    Returns
    -------
    usage: dict
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
    """
    Compute the total number of parameters of a network
    Parameters
    ----------
    net (nn.Module): the network

    Returns
    -------
    int: the number of parameters
    """
    num_parameters = 0
    for name, param in net.named_parameters():
        num_parameters += param.clone().cpu().data.view(-1).size(0)

    return num_parameters


#####################
# Indexed functions #
#####################

def create_index_matrix(nbRow, nbCol, injTable):
    """
    Creates the matrix of index of the pixels of the vector images to convert hexagonal images to square ones.
    Parameters
    ----------
    nbRow (int): the number of rows of the index matrix
    nbCol (int): the number of cols of the index matrix
    injTable (np.array): the list of the index of the pixels of vector hexagonal image in a vector square image
    eg : hexagonal image : [1, 2, 3, 4, 5, 6]
    injTable : [3, 25, 26, 58, 59, 60]
    square image : [0, 0, 1, ...., 0, 2, 3, 0, ...., 0, 4, 5, 6, 0, ...]

    Returns
    -------
    A Variable
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
    """
     Transforms a batch of features of vector images in a batch of features of matrix images
     Parameters
     ----------
     input_images (torch.Tensor): torch Tensor of images with shape (batch, features, image)

     Returns
     -------
     Batch of features of matrix images
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
    Transforms a batch of features of matrix images in a batch of features of vector images
    Parameters
    ----------
    input_matrix (torch.Tensor): Variable(torch Tensor) of images with shape (batch, features, matrix.size)
    index_matrix (torch.Tensor): matrix of index for the images, shape(1, 1, matrix.size)

    Returns
    -------

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
    Pools an index matrix
    Parameters
    ----------
    index_matrix (torch.Tensor): matrix of index for the images, shape(1, 1, matrix.size)
    kernel_type (str): the kernel shape, Hex for hexagonal Square for a square of size 3 and Pool for a square of size 2
    stride (int): the stride

    Returns
    -------
    The pooled matrix
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


def neighbours_extraction(index_matrix, kernel_type='Hex', stride=1):
    """

    Parameters
    ----------
    index_matrix (torch.Tensor): matrix of index for the images, shape(1, 1, matrix.size)
    kernel_type (str): the kernel shape, Hex for hexagonal Square for a square of size 3 and Pool for a square of size 2
    stride (int): the stride

    Returns
    -------

    """
    # TODO Find an more elegant solution to compute padding
    logger = logging.getLogger(__name__ + '.neighbours_extraction')
    padding = 2
    stride = stride
    bound = 2
    if kernel_type == 'Pool':
        kernel = np.ones((2, 2), dtype=bool)
        kernel_size = 2
        stride = 2
        bound = 1
        padding = 0
    elif kernel_type == 'One':
        kernel = np.ones((1, 1), dtype=bool)
        kernel_size = 1
        bound = 1
        padding = 0
    elif kernel_type == 'Hex_2':
        kernel = np.ones((5, 5), dtype=bool)
        kernel[0, 3:5] = False
        kernel[1, 4] = False
        kernel[3:5, 0] = False
        kernel[4, 1] = False
        kernel_size = 5
        padding = 4
        bound = 4
    else:
        kernel = np.ones((3, 3), dtype=bool)
        kernel_size = 3
        if kernel_type == 'Hex':
            kernel[0, 2] = False
            kernel[2, 0] = False

    neighbours = []
    try:
        assert padding % 2 == 0
    except AssertionError as err:
        logger.exception('Padding must be a multiple of 2 but is {}'.format(padding))
        raise err

    idx_mtx = np.ones((index_matrix.size(-2)+padding, index_matrix.size(-1)+padding), dtype=int) * (-1)
    offset = int(padding/2)
    if offset == 0:
        idx_mtx = index_matrix[0, 0, :, :].numpy()
    else:
        idx_mtx[offset:-offset, offset:-offset] = index_matrix[0, 0, :, :].numpy()

    for i in range(0, idx_mtx.shape[0]-bound, stride):
        for j in range(0, idx_mtx.shape[1]-bound, stride):
            patch = idx_mtx[i:i+kernel_size, j:j+kernel_size][kernel]
            if (kernel_type == 'Hex') and (patch[3] == -1):
                continue
            elif (kernel_type == 'Square') and (patch[4] == -1):
                continue
            elif (kernel_type == 'Pool') and (patch[0] == -1):
                continue
            elif (kernel_type == 'Hex_2') and (patch[9] == -1):
                continue
            elif (kernel_type == 'One') and (patch[0] == -1):
                continue
            neighbours.append(patch)

    neighbours = np.asarray(neighbours).T
    neighbours = torch.from_numpy(neighbours).long()

    return neighbours


def prepare_mask(indices):
    """
    Function to prepare the indices and the mask for the gemm im2col operation
    Parameters
    ----------
    indices

    Returns
    -------

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

def square_to_hexagonal_basic(image):
    """
    Rough sampling of square images to hexagonal grid
    Parameters
    ----------
    image (torch.Tensor of shape (c, n, m)

    Returns
    -------
    the image as a torch.Tensor and its index matrix
    """
    index_matrix = torch.ones(image.shape[1], image.shape[2] + np.ceil(image.shape[1]/2)) * -1
    image_vec = torch.zeros((image.shape[0], image.shape[1] * image.shape[2]))
    n = 0
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            if i % 2 == 1:
                if j == (image.shape[2] - 1):
                    image_vec[:, n] = image[:, i, j]
                else:
                    image_vec[:, n] = (image[:, i, j] + image[:, i, j+1]) / 2
            else:
                image_vec[:, n] = image[:, i, j]
            index_matrix[i, j + int(np.ceil(i/2))] = n
            n += 1
    # image_vec = torch.Tensor(image_vec)
    return image_vec, index_matrix


def square_to_hexagonal_index_matrix(image):
    """
        Creates the index matrix of square images in a hexagonal grid (axial)
        Parameters
        ----------
        image (torch.Tensor of shape (c, n, m)

        Returns
        -------
        index matrix
        """
    index_matrix = torch.ones(image.shape[1], image.shape[2] + np.ceil(image.shape[1] / 2)) * -1
    n = 0
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            index_matrix[i, j + int(np.ceil(i / 2))] = n
            n += 1
    return index_matrix


def square_to_hexagonal(image):
    """
    Rough sampling of square images to hexagonal grid
    Parameters
    ----------
    image (torch.Tensor of shape (c, n, m)

    Returns
    -------
    the image as a torch.Tensor
    """
    image_tr = image.clone()
    image_tr[:, :, :-1] = image[:, :, 1:]
    image_mean = (image + image_tr) / 2
    image[:, 1::2, :] = image_mean[:, 1::2, :]

    return image.view(image.shape[0], -1)


def build_hexagonal_position(index_matrix):
    """

    Parameters
    ----------
    index_matrix

    Returns
    -------

    """
    pix_positions = []
    for i in range(index_matrix.shape[0]):
        for j in range(index_matrix.shape[1]):
            if not index_matrix[i, j] == -1:
                pix_positions.append([j - i/2, -i])
    return pix_positions


def normalize(images):
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
        The covariance is C = 1/(n-1) * D * D.T
        The eigendecomp of C is: C = V Sigma V.T
        Let Y = 1/sqrt(n-1) * D
        Let U S V = svd(Y),
        Then the columns of U are the eigenvectors of:
        Y * Y.T = C
        And the singular values S are the sqrts of the eigenvalues of C
        We can apply PCA by multiplying by U.T
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

    def transform(self, D, whiten=False, ZCA=False,
                  regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by Sigma^(-1/2) U.T
        Any orthogonal transformation of this is also white,
        and when ZCA=True we choose:
         U Sigma^(-1/2) U.T
        """
        if whiten:
            # Compute Sigma^(-1/2) = S^-1,
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
    """Convert a numpy array to a tensor"""

    def __call__(self, data):

        data = torch.tensor(data)
        return data


class SquareToHexa(object):
    def __call__(self, image):
        image = square_to_hexagonal(image)
        # print(sample)
        return image
