import unittest

import torch

from indexedconv.engine import MaskedConv2d


class TestMaskedConv(unittest.TestCase):

    def setUp(self):
        self.data = torch.tensor([[1, 0, 2, 0, 3],
                                  [0, 4, 0, 5, 1],
                                  [0, 2, 0, 3, 0],
                                  [4, 0, 5, 1, 0],
                                  [2, 0, 3, 0, 4]],
                                 dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.convk3 = MaskedConv2d(1, 1, kernel_size=3, padding=1)
        self.convk3.bias = torch.nn.Parameter(torch.tensor([1.]))
        self.convk3.weight = torch.nn.Parameter(torch.tensor([[[[-1, -1, -1],
                                                              [-1, 8, -1],
                                                              [-1, -1, -1]]]], dtype=torch.float))
        self.convk5 = MaskedConv2d(1, 1, kernel_size=5, padding=2)
        self.convk5.bias = torch.nn.Parameter(torch.tensor([1.]))
        self.convk5.weight = torch.nn.Parameter(torch.tensor([[[[-1, -1, -1, -1, -1],
                                                              [-1, -1, -1, -1, -1],
                                                              [-1, -1, 24, -1, -1],
                                                              [-1, -1, -1, -1, -1],
                                                              [-1, -1, -1, -1, -1]]]], dtype=torch.float))

    def test_maskedconvk3(self):
        torch.testing.assert_allclose(self.convk3(self.data),
                                      torch.tensor([[5, -6, 12, -10, 24],
                                                    [-6, 30, -13, 35, 1],
                                                    [-5, 8, -14, 19, -8],
                                                    [31, -13, 35, -3, -7],
                                                    [13, -8, 20, -12, 32]], dtype=torch.float).unsqueeze(0).unsqueeze(0))

    def test_maskedconvk5(self):
        torch.testing.assert_allclose(self.convk5(self.data),
                                      torch.tensor([[17.,  -16.,   32.,  -13.,   65.],
                                                    [-15.,   78.,  -23.,  107.,   12.],
                                                    [-20.,   28.,  -29.,   49.,  -18.],
                                                    [85.,  -20.,  100.,   -1.,  -18.],
                                                    [42.,  -15.,   55.,  -17.,   85.]], dtype=torch.float).unsqueeze(0))


if __name__ == '__main__':
    unittest.main()
