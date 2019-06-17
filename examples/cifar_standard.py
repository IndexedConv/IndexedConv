import logging
import os
import sys
import argparse

import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import indexedconv.utils as utils
from indexedconv.nets.cifar import WideNet


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


if __name__ == '__main__':

    description = 'A ResNet like network is trained for a classification task on the CIFAR10 dataset.'
    # Parse script arguments
    print('parse arguments')
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("main_directory", help="path to the main directory of the experiments")
    parser.add_argument("data_directory", help="path to the data directory")
    parser.add_argument("exp_name", help="name of the experiment")
    parser.add_argument('--batch', help='batch size', type=int, default=125)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=300)
    parser.add_argument('--seeds', nargs='+', help='seeds to use, one training per seed', type=int,
                        default=range(1, 11))
    parser.add_argument('--device', help='device to use, for example cpu or cuda:0', type=str, default='cuda:0')

    args = parser.parse_args()

    main_directory = args.main_directory
    data_directory = args.data_directory
    experiment_name = args.exp_name
    batch_size = args.batch
    max_epochs = args.epochs
    seeds = args.seeds
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

    # Experiment parameters
    logger.info('batch_size : {}'.format(batch_size))
    logger.info('max_epochs : {}'.format(max_epochs))
    logger.info('cuda available : {}'.format(torch.cuda.is_available()))

    # Data
    logger.info('Original CIFAR')
    train_set = datasets.CIFAR10(data_directory, train=True, download=True)
    train_data_all = train_set.train_data.transpose(0, 3, 1, 2).astype(np.float32)

    original_size = train_data_all.shape[2:]

    train_data_all = train_data_all.reshape(train_data_all.shape[0], train_data_all.shape[1], -1)
    train_labels_all = train_set.train_labels
    test_set = datasets.CIFAR10(data_directory, train=False)
    test_data = test_set.test_data.transpose(0, 3, 1, 2).astype(np.float32)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], -1)
    test_labels = test_set.test_labels

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
    train_data_all = train_data_all_flat.T.reshape(train_data_all.shape[0:2] + original_size)

    test_data_flat = pca_all.transform(D=test_data_flat, whiten=True, ZCA=True)
    test_data = test_data_flat.T.reshape(test_data.shape[0:2] + original_size)

    # Datasets
    train_set = utils.NumpyDataset(train_data_all, train_labels_all, transform=utils.NumpyToTensor())
    val_set = utils.NumpyDataset(test_data, test_labels, transform=utils.NumpyToTensor())
    test_set = utils.NumpyDataset(test_data, test_labels, transform=utils.NumpyToTensor())
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # Run the experiments
    for seed in seeds:
        logger.info('Train model with seed {}'.format(seed))
        # TensorboardX writer
        writer = SummaryWriter(main_directory + '/runs/' + experiment_name + '_' + str(seed))

        # The model
        torch.manual_seed(seed)
        model = WideNet().to(device)
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
        # test(model, device, test_loader, epoch, val=False)

        writer.close()
