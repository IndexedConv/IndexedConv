import logging
import os
import sys

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


main_directory = '.'
experiment_name = 'Deep_mnist_hexa'
experiment_directory = main_directory + '/' + experiment_name
if not os.path.exists(experiment_directory):
    os.makedirs(experiment_directory)

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

batch_size = 64
test_batch_size = 1000

train_set = datasets.MNIST(main_directory + '/../ext_data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                               utils.SquareToHexa()
                           ]))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(main_directory + '/../ext_data', train=False,
                                                         transform=transforms.Compose([
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.1307,), (0.3081,)),
                                                             utils.SquareToHexa()
                                                         ])),
                                          batch_size=test_batch_size, shuffle=True)

device = torch.device("cuda")

# Plot a resampled image to check
img, _ = datasets.MNIST(main_directory + '/../ext_data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))[9]
# print(img.shape)
vec, index_matrix = utils.square_to_hexagonal(img)

# hex_mat = torch.zeros(index_matrix.shape)
# for i in range(index_matrix.shape[0]):
#     for j in range(index_matrix.shape[1]):
#         if not int(index_matrix[i, j]) == -1:
#             hex_mat[i, j] = vec[int(index_matrix[i, j])]
#
# print(hex_mat)

# The Deep MNIST model
model = GLNet2HexaConvForMnist(index_matrix).to(device)

optimizer = optim.Adam(model.parameters())

# Train and test
for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
