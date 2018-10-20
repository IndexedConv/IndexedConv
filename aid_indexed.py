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
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import utils.utils as utils
from utils.data import NumpyDataset, NumpyToTensor, HDF5Dataset, SquareToHexa
from nets.aid import WideNetIndexConvIndexPool


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
        if batch_idx % 20 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
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

    test_loss /= len(test_loader.sampler)
    accuracy = 100. * correct / len(test_loader.sampler)
    if val:
        logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.sampler), accuracy))
        if writer:
            writer.add_scalars('Loss', {'validating': test_loss}, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)
    else:
        logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.sampler), accuracy))


def plot_image(img, img_hex, index_matrix, path, writer=None):

    idx_mtx = index_matrix.view(index_matrix.shape[-2:])
    pix_pos = utils.build_hexagonal_position(idx_mtx)

    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(img.transpose(0, 2).transpose(0, 1))
    ax0.set_aspect('equal')
    ax0.set_axis_off()
    ax1.scatter(list(map(lambda x: x[0], pix_pos)), list(map(lambda x: x[1], pix_pos)), s=25,
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
    experiment_name = 'IndexedConv_aid_indexed'
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
    batch_size = 100
    test_batch_size = 1000
    max_epochs = 300
    resize_size = (64, 64)
    validating_ratio = 0.2
    logger.info('batch_size : {}'.format(batch_size))
    logger.info('test_batch_size : {}'.format(test_batch_size))
    logger.info('max_epochs : {}'.format(max_epochs))
    seeds = range(1, 11)

    hexa = False

    device = torch.device("cuda:0")
    logger.info('cuda available : {}'.format(torch.cuda.is_available()))

    # Data
    if hexa:
        camera_layout = 'Hex'
        logger.info('Hexagonal AID')

        if not os.path.exists(data_directory + '/aid' + str(resize_size[0]) + '_hexa.h5'):
            logger.info('Create hexagonal AID dataset')
            img, _ = datasets.ImageFolder(data_directory + '/AID',
                                          transform=transforms.Compose([transforms.Resize(resize_size),
                                                                        transforms.ToTensor()]))[0]
            index_matrix = utils.square_to_hexagonal_index_matrix(img)
            aid = datasets.ImageFolder(data_directory + '/AID',
                                       transform=transforms.Compose([transforms.Resize(resize_size),
                                                                     transforms.ToTensor(),
                                                                     SquareToHexa()]))
            with h5py.File(data_directory + '/aid' + str(resize_size[0]) + '_hexa.h5', 'w') as f:
                images = []
                labels = []
                for i in range(len(aid)):
                    image, label = aid[i]
                    images.append(image.numpy())
                    labels.append(label)
                f.create_dataset('images', data=np.array(images))
                f.create_dataset('labels', data=np.array(labels))
                f.create_dataset('index_matrix', data= index_matrix)
                f.attrs['class_names'] = np.array(aid.classes, dtype=h5py.special_dtype(vlen=str))

        # load hexagonal cifar
        f = h5py.File(data_directory + '/aid' + str(resize_size[0]) + '_hexa.h5', 'r')
        data = f['images'][()]
        labels = f['labels'][()]
        index_matrix = torch.tensor(f['index_matrix'][()])
        class_names = f.attrs['class_names']
        f.close()

    else:
        camera_layout = 'Square'
        logger.info('Square AID')

        if not os.path.exists(data_directory + '/aid' + str(resize_size[0]) + '.h5'):
            logger.info('Create resized AID dataset')
            aid = datasets.ImageFolder(data_directory + '/AID',
                                       transform=transforms.Compose([transforms.Resize(resize_size),
                                                                     transforms.ToTensor()]))
            index_matrix = torch.arange(aid[0][0].shape[1] * aid[0][0].shape[2])
            index_matrix = index_matrix.reshape(aid[0][0].shape[1], aid[0][0].shape[2])
            with h5py.File(data_directory + '/aid' + str(resize_size[0]) + '.h5', 'w') as f:
                images = []
                labels = []
                for i in range(len(aid)):
                    image, label = aid[i]
                    images.append(image.view(image.shape[0], -1).numpy())
                    labels.append(label)
                f.create_dataset('images', data=np.array(images))
                f.create_dataset('labels', data=np.array(labels))
                f.create_dataset('index_matrix', data=index_matrix)
                f.attrs['class_names'] = np.array(aid.classes, dtype=h5py.special_dtype(vlen=str))

        # load square aid
        f = h5py.File(data_directory + '/aid' + str(resize_size[0]) + '.h5', 'r')
        data = f['images'][()]
        labels = f['labels'][()]
        index_matrix = torch.tensor(f['index_matrix'][()])
        class_names = f.attrs['class_names']
        f.close()

    index_matrix.unsqueeze_(0)
    index_matrix.unsqueeze_(0)

    # Normalize data
    data = utils.normalize(data)

    # Datasets
    dataset = NumpyDataset(data, labels, transform=NumpyToTensor())

    # Run the experiments
    for seed in seeds:
        # Data loaders
        logger.info('Split data with seed {}'.format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_indices = []
        val_indices = []
        for cls in np.unique(labels):
            indices = np.where(labels == cls)
            indices = np.random.permutation(indices[0])
            train_indices.append(indices[:int(len(indices) * (1 - validating_ratio))])
            val_indices.append(indices[int(len(indices) * (1 - validating_ratio)):])
        train_set_sampler = sampler.SubsetRandomSampler(np.concatenate(train_indices))
        validating_set_sampler = sampler.SubsetRandomSampler(np.concatenate(val_indices))
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_set_sampler, num_workers=8)
        val_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=validating_set_sampler, num_workers=8)

        # TensorboardX writer
        writer = SummaryWriter(main_directory + '/runs/' + experiment_name + '_' + str(seed))

        # Plot a resampled image to check
        if hexa:
            img, _ = datasets.ImageFolder(data_directory + '/AID',
                                          transform=transforms.Compose([transforms.Resize(resize_size),
                                                                        transforms.ToTensor()]))[seed]
            img_hex, _ = HDF5Dataset(data_directory + '/aid' + str(resize_size[0]) + '_hexa.h5',
                                     transform=NumpyToTensor())[seed]
            plot_image(img, img_hex, index_matrix,
                       experiment_directory + '/hex_aid_' + str(seed) + '.png', writer=writer)

        # The model
        torch.manual_seed(0)
        model = WideNetIndexConvIndexPool(index_matrix, camera_layout, len(class_names)).to(device)
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
