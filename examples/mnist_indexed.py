import logging
import sys
import argparse

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

import indexedconv.utils as utils
from indexedconv.nets.mnist import GLNet2HexaConvForMnist


class ToVector(object):
    def __call__(self, sample):
        sample = sample.view(1, -1)
        return sample


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        logger.info('loss : {}'.format(loss))
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
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
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    description = 'Demonstration of the use of IndexedConv package. A convnet is trained ' \
                  'for a classification task on the MNIST dataset.'
    # Parse script arguments
    print('parse arguments')
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument('main_directory', help='path to the main directory of the experiments')
    parser.add_argument('data_directory', help='path to the data directory')
    parser.add_argument('exp_name', help='name of the experiment')
    parser.add_argument('--hexa', help='the pixel grid of the images. True for hexagonal, False for cartesian.',
                        action='store_true', default=False)
    parser.add_argument('--batch', help='batch size', type=int, default=64)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=10)
    parser.add_argument('--seed', nargs='+', help='seed to use', type=int, default=0)
    parser.add_argument('--device', help='device to use, for example cpu or cuda:0', type=str, default='cuda:0')

    args = parser.parse_args()

    main_directory = args.main_directory
    data_directory = args.data_directory
    experiment_name = args.exp_name
    hexa = args.hexa
    batch_size = args.batch
    max_epochs = args.epochs
    seed = args.seed
    device = torch.device(args.device)

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

    train_set = datasets.MNIST(data_directory, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   utils.SquareToHexa()
                               ]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_directory, train=False,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,)),
                                                                 utils.SquareToHexa()
                                                             ])),
                                              batch_size=batch_size, shuffle=True)

    # Compute index matrix
    img, _ = datasets.MNIST(data_directory, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))[0]
    _, index_matrix = utils.square_to_hexagonal(img)

    # The Deep MNIST model
    torch.manual_seed(seed)
    model = GLNet2HexaConvForMnist(index_matrix).to(device)

    optimizer = optim.Adam(model.parameters())

    # Train and test
    for epoch in range(1, max_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
