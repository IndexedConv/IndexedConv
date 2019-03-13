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

    positions = np.random.randn((40, 2)) * 100
    data = torch.rand(68, 40)

    # Convolution
    indices
