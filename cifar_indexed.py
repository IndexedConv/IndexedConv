import logging
import os
import sys
import h5py

import matplotlib as mpl
mpl.use('Agg')  # Because of an issue in Qt5 causing seg fault
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import indexedconv.utils.utils as utils
from indexedconv.utils.data import NumpyDataset, NumpyToTensor, HDF5Dataset, SquareToHexa
from indexedconv.nets.cifar import WideNetIndexConvIndexPool, WideNet


def train(model, device, train_loader, optimizer, epoch, writer=None):
    model.train()
    loss_values = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    if writer:
        writer.add_scalars('Loss', {'training': np.mean(loss_values)}, epoch)


def test(model, device, test_loader, epoch, val=True, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if val:
        logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))
        if writer:
            writer.add_scalars('Loss', {'validating': test_loss}, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)
    else:
        logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))


def plot_image(img, img_hex, index_matrix, path, writer=None):

    idx_mtx = index_matrix.view(index_matrix.shape[-2:])
    pix_pos = utils.build_hexagonal_position(idx_mtx)

    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(img.transpose(0, 2).transpose(0, 1))
    ax0.set_aspect('equal')
    ax0.set_axis_off()
    ax1.scatter(list(map(lambda x: x[1], pix_pos)), list(map(lambda x: x[0], pix_pos)), s=25,
                c=img_hex.transpose(1, 0).numpy(), marker=(6, 0, 0))
    ax1.set_aspect('equal')
    ax1.set_axis_off()
    plt.savefig(path)
    if writer:
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_square_vs_hexa = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        writer.add_image('Square_vs_Hexagonal', transforms.ToTensor()(pil_square_vs_hexa))


if __name__ == '__main__':
    main_directory = '.'
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
    experiment_name = 'IndexedConv_cifar_wideAxialnet'
    data_directory = main_directory + '/../ext_data'
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

    # Experiment parameters
    batch_size = 125
    test_batch_size = 1000
    max_epochs = 300
    logger.info('batch_size : {}'.format(batch_size))
    logger.info('test_batch_size : {}'.format(test_batch_size))
    logger.info('max_epochs : {}'.format(max_epochs))
    seeds = range(10, 11)

    hexa = True

    device = torch.device("cuda:0")
    logger.info('cuda available : {}'.format(torch.cuda.is_available()))

    # Data
    if hexa:
        camera_layout = 'Hex'
        logger.info('Hexagonal CIFAR')
        img, _ = datasets.CIFAR10(data_directory, train=True, download=True, transform=transforms.ToTensor())[0]
        index_matrix = utils.square_to_hexagonal_index_matrix(img)

        if not os.path.exists(data_directory + '/cifar10.hdf5'):
            train_set = datasets.CIFAR10(data_directory, train=True, download=True,
                                         transform=transforms.Compose([transforms.ToTensor(), SquareToHexa()]))
            with h5py.File(data_directory + '/cifar10.hdf5', 'w') as f:
                images = []
                labels = []
                for i in range(len(train_set)):
                    image, label = train_set[i]
                    images.append(image.numpy())
                    labels.append(label)
                f.create_dataset('images', data=np.array(images))
                f.create_dataset('labels', data=np.array(labels))
                f.attrs['index_matrix'] = index_matrix
        if not os.path.exists(data_directory + '/cifar10_test.hdf5'):
            test_set = datasets.CIFAR10(data_directory, train=False,
                                        transform=transforms.Compose([transforms.ToTensor(), SquareToHexa()]))
            with h5py.File(data_directory + '/cifar10_test.hdf5', 'w') as f:
                images = []
                labels = []
                for i in range(len(test_set)):
                    image, label = test_set[i]
                    images.append(image.numpy())
                    labels.append(label)
                f.create_dataset('images', data=np.array(images))
                f.create_dataset('labels', data=np.array(labels))
                f.attrs['index_matrix'] = index_matrix

        # load hexagonal cifar
        f_train = h5py.File(data_directory + '/cifar10.hdf5', 'r')
        train_data_all = f_train['images'][()]
        train_labels_all = f_train['labels'][()]
        f_train.close()
        f_test = h5py.File(data_directory + '/cifar10_test.hdf5', 'r')
        test_data = f_test['images'][()]
        test_labels = f_test['labels'][()]
        f_test.close()
    else:
        camera_layout = 'Square'
        logger.info('Original CIFAR')
        train_set = datasets.CIFAR10(data_directory, train=True, download=True)
        train_data_all = train_set.train_data.transpose(0, 3, 1, 2).astype(np.float32)
        index_matrix = torch.arange(train_data_all.shape[2] * train_data_all.shape[3])
        index_matrix = index_matrix.reshape(train_data_all.shape[2], train_data_all.shape[3])
        train_data_all = train_data_all.reshape(train_data_all.shape[0], train_data_all.shape[1], -1)
        train_labels_all = train_set.train_labels
        test_set = datasets.CIFAR10(data_directory, train=False)
        test_data = test_set.test_data.transpose(0, 3, 1, 2).astype(np.float32)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], -1)
        test_labels = test_set.test_labels

    index_matrix.unsqueeze_(0)
    index_matrix.unsqueeze_(0)

    # Normalize data
    train_data_all = utils.normalize(train_data_all)
    test_data = utils.normalize(test_data)

    # Whitening data
    logger.info('Computing whitening matrices...')
    train_data_all_flat = train_data_all.reshape(train_data_all.shape[0], -1).T
    test_data_flat = test_data.reshape(test_data.shape[0], -1).T
    pca_all = utils.PCA(D=train_data_all_flat, n_components=train_data_all_flat.shape[1])

    logger.info('Whitening data...')
    train_data_all_flat = pca_all.transform(D=train_data_all_flat, whiten=True, ZCA=True)
    train_data_all = train_data_all_flat.T.reshape(train_data_all.shape[0:2] + (-1,))

    test_data_flat = pca_all.transform(D=test_data_flat, whiten=True, ZCA=True)
    test_data = test_data_flat.T.reshape(test_data.shape[0:2] + (-1,))

    # Datasets
    train_set = NumpyDataset(train_data_all, train_labels_all, transform=NumpyToTensor())
    val_set = NumpyDataset(test_data, test_labels, transform=NumpyToTensor())
    test_set = NumpyDataset(test_data, test_labels, transform=NumpyToTensor())
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=8)

    # Run the experiments
    for seed in seeds:
        logger.info('Train model with seed {}'.format(seed))
        # TensorboardX writer
        writer = SummaryWriter(main_directory + '/runs/' + experiment_name + '_' + str(seed))

        # Plot a resampled image to check
        if hexa:
            img, _ = datasets.CIFAR10(data_directory, train=True, download=True, transform=transforms.ToTensor())[seed]
            img_hex, _ = HDF5Dataset(data_directory + '/cifar10.hdf5', transform=NumpyToTensor())[seed]
            plot_image(img, img_hex, index_matrix,
                       experiment_directory + '/hex_cifar_' + str(seed) + '.png', writer=writer)

        # The model
        torch.manual_seed(seed)
        model = WideNetIndexConvIndexPool(index_matrix, camera_layout).to(device)

        logger.info('Net parameters number : {}'.format(utils.compute_total_parameter_number(model)))

        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

        # Train and test
        logger.info('Start training')
        for epoch in range(1, max_epochs + 1):
            gpu_map = utils.get_gpu_usage_map(0)
            logger.info('GPU usage : {}'.format(gpu_map))
            train(model, device, train_loader, optimizer, epoch, writer=writer)
            test(model, device, val_loader, epoch, writer=writer)
            scheduler.step(epoch=epoch)
            if epoch % 100 == 0:
                torch.save(model.state_dict(), experiment_directory + '/model_' + str(seed) + '_epoch_' + str(epoch))

        writer.close()
