import os
import sys
import time
import logging
import h5py
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import indexedconv.utils as utils
import indexedconv.engine as engine
from indexedconv.nets.aid import WideNet, WideNetIndexConvIndexPool, WideNetMasked


if __name__ == '__main__':

    description = 'A ResNet like network is trained for a classification task on the CIFAR10 dataset.'
    # Parse script arguments
    print('parse arguments')
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument('main_directory', help='path to the main directory of the experiments')
    parser.add_argument('data_directory', help='path to the data directory')
    parser.add_argument('exp_name', help='name of the experiment')
    parser.add_argument('--iter', help='number of iterations', type=int, default=100)
    parser.add_argument('--size', help='image size', type=int, default=64)
    parser.add_argument('--cin', help='number of channels of input data', type=int, default=3)
    parser.add_argument('--cout', help='number of features of the convolution layer', type=int, default=32)
    parser.add_argument('--batch', nargs='+', type=int, default=[16, 32, 64, 100])
    parser.add_argument('--device', help='device to use, for example cpu or cuda:0', type=str, default='cuda:0')

    args = parser.parse_args()

    main_directory = args.main_directory
    data_directory = args.data_directory
    experiment_name = args.exp_name
    iterations = args.iter
    size = args.size
    c_in = args.cin
    c_out = args.cout
    batch_sizes = args.batch
    device = torch.device(args.device)

    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

    experiment_directory = main_directory + '/' + experiment_name
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    formatter_file = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    file_handler = logging.FileHandler('{}/{}/{}.log'.format(main_directory,
                                                             experiment_name,
                                                             experiment_name))
    file_handler.setFormatter(formatter_file)
    logger.addHandler(file_handler)

    indexed_conv = []
    nn_conv = []
    indexed_square_net = []
    nn_square_net = []
    indexed_hexa_net = []
    nn_hexa_net = []
    indexed_conv_ram = []
    nn_conv_ram = []
    indexed_square_net_ram = []
    nn_square_net_ram = []
    indexed_hexa_net_ram = []
    nn_hexa_net_ram = []

    logger.info('torch version : {}'.format(torch.__version__))
    for batch_size in batch_sizes:
        image_size = (batch_size, c_in, size, size)

        dummy_data = torch.rand(image_size)

        dummy_data = dummy_data.to(device)

        logger.info('Compare indexed conv and nn.Conv2d on square dummy images with 1 conv')
        logger.info('batch size: {} iterations: {}'.format(batch_size, iterations))

        cv_nn = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)

        index_matrix_square = torch.arange(image_size[2] * image_size[3]).view(image_size[2:]).unsqueeze(0).unsqueeze(0)
        indices_conv0_square = utils.neighbours_extraction(index_matrix_square,
                                                           kernel_type='Square',
                                                           stride=1)
        cv_square = engine.IndexedConv(c_in, c_out, indices_conv0_square)

        cv_nn.to(device)
        cv_square.to(device)
        ram_b = torch.cuda.memory_allocated() / 1024 / 1024
        start_square = time.time()
        for _ in range(iterations):
            cv_square.zero_grad()
            convoluted_square = cv_square(dummy_data.view(batch_size, c_in, -1))
            loss_square = torch.sum(convoluted_square)
            loss_square.backward()
        t = (time.time() - start_square) / iterations
        indexed_conv.append(t)
        indexed_conv_ram.append(torch.cuda.memory_allocated() / 1024 / 1024 - ram_b)
        logger.info('Time for 1 indexed conv + backward : {}'.format(t))
        del cv_square
        del convoluted_square
        torch.cuda.empty_cache()

        ram_b = torch.cuda.memory_allocated() / 1024 / 1024
        start_nn = time.time()
        for _ in range(iterations):
            cv_nn.zero_grad()
            convoluted_nn = cv_nn(dummy_data)
            loss_nn = torch.sum(convoluted_nn)
            loss_nn.backward()
        t = (time.time() - start_nn) / iterations
        nn_conv.append(t)
        nn_conv_ram.append(torch.cuda.memory_allocated() / 1024 / 1024 - ram_b)
        logger.info('Time for 1 nn conv + backward : {}'.format(t))
        del cv_nn
        del convoluted_nn
        torch.cuda.empty_cache()

        logger.info('Compare indexed conv and nn.Conv2d on square images with WideNet')
        logger.info('batch size: {} iterations: {}'.format(batch_size, iterations))

        indexed_net = WideNetIndexConvIndexPool(index_matrix_square.float(), 'Square', 30).to(device)
        nn_net = WideNet(30).to(device)

        ram_b = torch.cuda.memory_allocated() / 1024 / 1024
        start_indexed = time.time()
        for _ in range(iterations):
            out = indexed_net(dummy_data.view(batch_size, c_in, -1))
            loss = torch.sum(out)
            loss.backward()
        t = (time.time() - start_indexed) / iterations
        indexed_square_net.append(t)
        indexed_square_net_ram.append(torch.cuda.memory_allocated() / 1024 / 1024 - ram_b)
        logger.info('Time for indexed widenet {}'.format(t))
        del indexed_net
        del out
        torch.cuda.empty_cache()

        ram_b = torch.cuda.memory_allocated() / 1024 / 1024
        start_nn = time.time()
        for _ in range(iterations):
            out = nn_net(dummy_data)
            loss = torch.sum(out)
            loss.backward()
        t = (time.time() - start_nn) / iterations
        nn_square_net.append(t)
        nn_square_net_ram.append(torch.cuda.memory_allocated() / 1024 / 1024 - ram_b)
        logger.info('Time for nn widenet {}'.format(t))
        del nn_net
        del out
        torch.cuda.empty_cache()

        logger.info('Compare indexed conv and nn.Conv2d on hexagonal images with WideNet')

        f = h5py.File(data_directory + '/aid' + str(size) + '_hexa.h5', 'r')  # TODO check the existence of data
        data = f['images'][()]
        labels = f['labels'][()]
        index_matrix = torch.tensor(f['index_matrix'][()])
        class_names = f.attrs['class_names']
        f.close()

        data_shifted = np.zeros(data.shape[0:2] + index_matrix.shape).astype(np.float32)
        for i in range(index_matrix.shape[0]):
            for j in range(index_matrix.shape[1]):
                if not int(index_matrix[i, j]) == -1:
                    data_shifted[:, :, i, j] = data[:, :, int(index_matrix[i, j])]

        sh_dataset = utils.NumpyDataset(data_shifted, labels, transform=utils.NumpyToTensor())
        hex_dataset = utils.NumpyDataset(data, labels, transform=utils.NumpyToTensor())
        sh_loader = DataLoader(sh_dataset, batch_size=batch_size, shuffle=False)
        hex_loader = DataLoader(hex_dataset, batch_size=batch_size, shuffle=False)

        logger.info('batch size: {} iterations: {}'.format(batch_size, len(hex_loader)))

        index_matrix = index_matrix.unsqueeze_(0).unsqueeze_(0)
        indexed_net = WideNetIndexConvIndexPool(index_matrix, 'Hex', 30).to(device)
        nn_net = WideNetMasked(30).to(device)

        ram_b = torch.cuda.memory_allocated() / 1024 / 1024
        start_indexed = time.time()
        for d in hex_loader:
            out = indexed_net(d[0].to(device))
            loss = torch.sum(out)
            loss.backward()
        t = (time.time() - start_indexed) / len(hex_loader)
        indexed_hexa_net.append(t)
        indexed_hexa_net_ram.append(torch.cuda.memory_allocated() / 1024 / 1024 - ram_b)
        logger.info('Time for indexed widenet {}'.format(t))
        del indexed_net
        del out
        torch.cuda.empty_cache()

        ram_b = torch.cuda.memory_allocated() / 1024 / 1024
        start_nn = time.time()
        for d in sh_loader:
            out = nn_net(d[0].to(device))
            loss = torch.sum(out)
            loss.backward()
        t = (time.time() - start_nn) / len(sh_loader)
        nn_hexa_net.append(t)
        nn_hexa_net_ram.append(torch.cuda.memory_allocated() / 1024 / 1024 - ram_b)
        logger.info('Time for masked nn widenet {}'.format(t))
        del nn_net
        del out
        torch.cuda.empty_cache()

    dataf = pd.DataFrame()
    dataf['indexed_conv'] = indexed_conv
    dataf['nn_conv'] = nn_conv
    dataf['indexed_square_net'] = indexed_square_net
    dataf['nn_square_net'] = nn_square_net
    dataf['indexed_hexa_net'] = indexed_hexa_net
    dataf['nn_hexa_net'] = nn_hexa_net
    dataf['indexed_conv_ram'] = indexed_conv_ram
    dataf['nn_conv_ram'] = nn_conv_ram
    dataf['indexed_square_net_ram'] = indexed_square_net_ram
    dataf['nn_square_net_ram'] = nn_square_net_ram
    dataf['indexed_hexa_net_ram'] = indexed_hexa_net_ram
    dataf['nn_hexa_net_ram'] = nn_hexa_net_ram
    if device.type == 'gpu':
        device_name = torch.cuda.get_device_name(device.index).split(' ')[-1]
    else:
        device_name = ''
    dataf.to_hdf(main_directory + '/' + experiment_name + '/' + experiment_name + device_name + '.h5', key='data')
