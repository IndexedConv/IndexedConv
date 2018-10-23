import unittest

import torch

from modules.indexed import IndexedConv, IndexedAveragePool2d, IndexedMaxPool2d
from utils.utils import neighbours_extraction


class TestIndexedConv(unittest.TestCase):

    def setUp(self):
        self.data_1 = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.data_3 = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.float).unsqueeze(0)
        index_matrix = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).unsqueeze(0).unsqueeze(0)
        neighbours_indices = neighbours_extraction(index_matrix, 'Square')
        self.conv11 = IndexedConv(1, 1, neighbours_indices)
        self.conv11.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float))
        self.conv11.weight = torch.nn.Parameter(torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float))
        self.conv13 = IndexedConv(1, 3, neighbours_indices)
        self.conv13.bias = torch.nn.Parameter(torch.tensor([1, 1, 1], dtype=torch.float))
        self.conv13.weight = torch.nn.Parameter(torch.tensor([[-1, -1, -1, -1, 8, -1, -1, -1, -1],
                                                              [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                                                              [-1, -1, -1, -1, 8, -1, -1, -1, -1]], dtype=torch.float))
        self.conv31 = IndexedConv(3, 1, neighbours_indices)
        self.conv31.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float))
        self.conv31.weight = torch.nn.Parameter(torch.tensor([[-1, -1, -1, -1, 8, -1, -1, -1, -1],
                                                              [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                                                              [-1, -1, -1, -1, 8, -1, -1, -1, -1]], dtype=torch.float))

    def test_indexedconv11(self):
        torch.testing.assert_allclose(self.conv11(self.data_1),
                                      torch.tensor([8, -2, 8, -2, 5, -2, 8, -2, 8], dtype=torch.float).unsqueeze(0).unsqueeze(0))

    def test_indexedconv13(self):
        torch.testing.assert_allclose(self.conv13(self.data_1),
                                      torch.tensor([[8, -2, 8, -2, 5, -2, 8, -2, 8],
                                                    [8, -2, 8, -2, 5, -2, 8, -2, 8],
                                                    [8, -2, 8, -2, 5, -2, 8, -2, 8]], dtype=torch.float).unsqueeze(0))

    def test_indexedconv31(self):
        torch.testing.assert_allclose(self.conv31(self.data_3),
                                      torch.tensor([22, -8, 22, -8, 13, -8, 22, -8, 22], dtype=torch.float).unsqueeze(0).unsqueeze(0))


class TestIndexedMaxPool2d(unittest.TestCase):
    def setUp(self):
        self.data_1 = torch.tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4],
                                   dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.data_3 = torch.tensor([[1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4],
                                    [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4],
                                    [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4]],
                                   dtype=torch.float).unsqueeze(0)
        index_matrix = torch.tensor([[0, 1, 2, 3, 4],
                                     [5, 6, 7, 8, 9],
                                     [10, 11, 12, 13, 14],
                                     [15, 16, 17, 18, 19],
                                     [20, 21, 22, 23, 24]]).unsqueeze(0).unsqueeze(0)
        neighbours_indices = neighbours_extraction(index_matrix, kernel_type='Pool', stride=2)
        self.maxpool = IndexedMaxPool2d(neighbours_indices)

    def test_maxpool1(self):
        torch.testing.assert_allclose(self.maxpool(self.data_1),
                                      torch.tensor([[[4.,  5.,  4.,  5.]]]))

    def test_maxpool3(self):
        torch.testing.assert_allclose(self.maxpool(self.data_3),
                                      torch.tensor([[[4., 5., 4., 5.],
                                                     [4., 5., 4., 5.],
                                                     [4., 5., 4., 5.]]]))


class TestIndexedAveragePool2d(unittest.TestCase):
    def setUp(self):
        self.data_1 = torch.tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4],
                                   dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.data_3 = torch.tensor([[1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4],
                                    [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4],
                                    [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 0, 2, 0, 3, 0, 4]],
                                   dtype=torch.float).unsqueeze(0)
        index_matrix = torch.tensor([[0, 1, 2, 3, 4],
                                     [5, 6, 7, 8, 9],
                                     [10, 11, 12, 13, 14],
                                     [15, 16, 17, 18, 19],
                                     [20, 21, 22, 23, 24]]).unsqueeze(0).unsqueeze(0)
        neighbours_indices = neighbours_extraction(index_matrix, kernel_type='Pool', stride=2)
        self.maxpool = IndexedMaxPool2d(neighbours_indices)

    def test_maxpool1(self):
        torch.testing.assert_allclose(self.maxpool(self.data_1),
                                      torch.tensor([[[4.,  5.,  4.,  5.]]]))

    def test_maxpool3(self):
        torch.testing.assert_allclose(self.maxpool(self.data_3),
                                      torch.tensor([[[4., 5., 4., 5.],
                                                     [4., 5., 4., 5.],
                                                     [4., 5., 4., 5.]]]))


if __name__ == '__main__':
    unittest.main()

