import torch
from torch.utils.data import Dataset
import utils.utils as utils


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
        image = utils.square_to_hexagonal(image)
        # print(sample)
        return image

