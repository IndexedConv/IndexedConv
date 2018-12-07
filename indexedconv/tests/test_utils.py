import unittest

import torch
import numpy as np

import indexedconv.utils as utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.injunction_table = [0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22, 23, 24]
        self.indexed_matrix = torch.tensor([[[[0, 1, 2, -1, -1],
                                            [3, 4, 5, 6, -1],
                                            [7, 8, 9, 10, 11],
                                            [-1, 12, 13, 14, 15],
                                            [-1, -1, 16, 17, 18]
                                           ]]], dtype=torch.float)
        self.nbRow = 5
        self.nbCol = 5
        self.image = torch.tensor([[[1, 1, 1,
                                     2, 2, 2, 2,
                                     3, 3, 3, 3, 3,
                                     4, 4, 4, 4,
                                     5, 5, 5]]], dtype=torch.float)
        self.image_matrix = torch.tensor([[[[1, 1, 1, 0, 0],
                                            [2, 2, 2, 2, 0],
                                            [3, 3, 3, 3, 3],
                                            [0, 4, 4, 4, 4],
                                            [0, 0, 5, 5, 5]
                                            ]]], dtype=torch.float)
        self.pooled_index_matrix_pool = torch.tensor([[[[0, 1],
                                                        [2, 3]]]], dtype=torch.float)
        self.pooled_index_matrix_hex = torch.tensor([[[[0, 1, -1],
                                                       [2, 3, 4],
                                                       [-1, 5, 6]]]], dtype=torch.float)
        self.neighbours_indices_hex = torch.tensor(
            [[-1, -1, -1, -1, 0,  1,  2, -1,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14],
             [-1, -1, -1,  0, 1,  2, -1,  3,  4,  5,  6, -1,  8,  9, 10, 11, 13, 14, 15],
             [-1,  0,  1, -1, 3,  4,  5, -1,  7,  8,  9, 10, -1, 12, 13, 14, -1, 16, 17],
             [0,   1,  2,  3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
             [1,   2, -1,  4, 5,  6, -1,  8,  9, 10, 11, -1, 13, 14, 15, -1, 17, 18, -1],
             [3,   4,  5,  7, 8,  9, 10, -1, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1],
             [4,   5,  6,  8, 9, 10, 11, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1]],
            dtype=torch.float)
        self.neighbours_indices_pool = torch.tensor([[0, 2, 7, 9],
                                                    [1, -1, 8, 10],
                                                    [3, 5, -1, 13],
                                                    [4, 6, 12, 14]],
                                                    dtype=torch.float)
        self.neighbours_indices_hex_dilated = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  2, -1,  3,  4,  5,  7,  8,  9],
             [-1, -1, -1, -1, -1, -1, -1,  0,  1,  2, -1, -1,  4,  5,  6, -1,  9, 10, 11],
             [-1, -1,  0, -1, -1,  3,  4, -1, -1,  7,  8,  9, -1, -1, 12, 13, -1, -1, 16],
             [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
             [2,  -1, -1,  5,  6, -1, -1,  9, 10, 11, -1, -1, 14, 15, -1, -1, 18, -1, -1],
             [7,   8,  9, -1, 12, 13, 14, -1, -1, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1],
             [9,  10, 11, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.float)
        self.neighbours_indices_one = torch.tensor([[0, 2, 7, 9]], dtype=torch.float)
        self.neighbours_indices_retina_hex_3 = torch.tensor(
            [[-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  2, -1,  3,  4,  7,  9, -1, 14],
             [-1, -1, -1, -1, -1, -1, -1, -1,  0,  1,  2, -1, -1,  4,  5,  8, 10, 12, 15],
             [-1, -1, -1, -1, -1, -1, -1,  0,  1,  2, -1, -1, -1,  5,  6,  9, 11, 13, -1],
             [-1, -1, -1, -1, -1,  0, -1, -1, -1,  3,  4,  5,  6,  7,  8, -1, 13, -1, 17],
             [-1, -1, -1, -1,  0,  1, -1, -1,  3,  4,  5,  6, -1,  8,  9, 12, 14, -1, 18],
             [-1, -1, -1, -1,  1,  2, -1,  3,  4,  5,  6, -1, -1,  9, 10, 13, 15, 16, -1],
             [-1, -1, -1, -1,  2, -1,  3,  4,  5,  6, -1, -1, -1, 10, 11, 14, -1, 17, -1],
             [-1, -1, -1,  0, -1,  3, -1, -1, -1,  7,  8,  9, 10, -1, 12, -1, 16, -1, -1],
             [-1, -1, -1,  1,  3,  4, -1, -1,  7,  8,  9, 10, 11, 12, 13, -1, 17, -1, -1],
             [-1, -1,  0,  2,  4,  5, -1,  7,  8,  9, 10, 11, -1, 13, 14, 16, 18, -1, -1],
             [-1, -1,  1, -1,  5,  6,  7,  8,  9, 10, 11, -1, -1, 14, 15, 17, -1, -1, -1],
             [-1, -1,  2, -1,  6, -1,  8,  9, 10, 11, -1, -1, -1, 15, -1, 18, -1, -1, -1],
             [-1,  1, -1,  4,  7,  8, -1, -1, -1, 12, 13, 14, 15, -1, 16, -1, -1, -1, -1],
             [-1,  2,  3,  5,  8,  9, -1, -1, 12, 13, 14, 15, -1, 16, 17, -1, -1, -1, -1],
             [ 0, -1,  4,  6,  9, 10, -1, 12, 13, 14, 15, -1, -1, 17, 18, -1, -1, -1, -1],
             [ 1, -1,  5, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, 18, -1, -1, -1, -1, -1],
             [-1,  5,  7,  9, 12, 13, -1, -1, -1, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1],
             [ 3,  6,  8, 10, 13, 14, -1, -1, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1],
             [ 4, -1,  9, 11, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.float).transpose(1, 0)
        self.square_image = torch.tensor([[[1, 0, 1, 0, 1],
                                           [0, 1, 0, 1, 0],
                                           [1, 0, 1, 0, 1],
                                           [0, 1, 0, 1, 0],
                                           [1, 0, 1, 0, 1]
                                           ]], dtype=torch.float)
        self.square_image_hex = torch.tensor([[1, 0, 1, 0, 1,
                                               0.5, 0.5, 0.5, 0.5, 0,
                                               1, 0, 1, 0, 1,
                                               0.5, 0.5, 0.5, 0.5, 0,
                                               1, 0, 1, 0, 1
                                               ]], dtype=torch.float)
        self.square_to_hexagonal_index_matrix = torch.tensor([[0., 1., 2., 3., 4., -1., -1., -1.],
                                                              [-1., 5., 6., 7., 8., 9., -1., -1.],
                                                              [-1., 10., 11., 12., 13., 14., -1., -1.],
                                                              [-1., -1., 15., 16., 17., 18., 19., -1.],
                                                              [-1., -1., 20., 21., 22., 23., 24., -1.]])
        self.square_to_hexagonal_positions = [[0.0, 0], [1.0, 0], [2.0, 0], [3.0, 0], [4.0, 0],
                                              [0.5, -1], [1.5, -1], [2.5, -1], [3.5, -1], [4.5, -1],
                                              [0.0, -2], [1.0, -2], [2.0, -2], [3.0, -2], [4.0, -2],
                                              [0.5, -3], [1.5, -3], [2.5, -3], [3.5, -3], [4.5, -3],
                                              [0.0, -4], [1.0, -4], [2.0, -4], [3.0, -4], [4.0, -4]]
        self.hexagonal_kernel_1 = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        self.hexagonal_kernel_1_2 = np.array([[1, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0],
                                              [1, 0, 1, 0, 1],
                                              [0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 1]])
        self.hexagonal_kernel_2 = np.array([[1, 1, 1, 0, 0],
                                            [1, 1, 1, 1, 0],
                                            [1, 1, 1, 1, 1],
                                            [0, 1, 1, 1, 1],
                                            [0, 0, 1, 1, 1]])
        self.square_kernel_1 = np.ones((3, 3), dtype=int)
        self.square_kernel_1_2 = np.array([[1, 0, 1, 0, 1],
                                           [0, 0, 0, 0, 0],
                                           [1, 0, 1, 0, 1],
                                           [0, 0, 0, 0, 0],
                                           [1, 0, 1, 0, 1]])
        self.square_kernel_0 = np.ones((1, 1), dtype=int)

    def test_create_index_matrix(self):
        torch.testing.assert_allclose(utils.create_index_matrix(self.nbRow, self.nbCol, self.injunction_table),
                                      self.indexed_matrix)

    def test_img2mat(self):
        torch.testing.assert_allclose(utils.img2mat(self.image, self.indexed_matrix).type(torch.float),
                                      self.image_matrix)

    def test_mat2img(self):
        torch.testing.assert_allclose(utils.mat2img(self.image_matrix, self.indexed_matrix),
                                      self.image)

    def test_pool_index_matrix(self):
        torch.testing.assert_allclose(utils.pool_index_matrix(self.indexed_matrix, kernel_type='Hex'),
                                      self.pooled_index_matrix_hex)
        torch.testing.assert_allclose(utils.pool_index_matrix(self.indexed_matrix, kernel_type='Pool'),
                                      self.pooled_index_matrix_pool)

    def test_neighbours_extraction(self):
        torch.testing.assert_allclose(utils.neighbours_extraction(self.indexed_matrix).type(torch.float),
                                      self.neighbours_indices_hex)
        torch.testing.assert_allclose(
            utils.neighbours_extraction(self.indexed_matrix, kernel_type='Hex', dilation=2).type(torch.float),
            self.neighbours_indices_hex_dilated)
        torch.testing.assert_allclose(
            utils.neighbours_extraction(self.indexed_matrix, kernel_type='Pool').type(torch.float),
            self.neighbours_indices_pool)
        torch.testing.assert_allclose(
            utils.neighbours_extraction(self.indexed_matrix, kernel_type='Square', radius=0, stride=2).type(torch.float),
            self.neighbours_indices_one)
        torch.testing.assert_allclose(
            utils.neighbours_extraction(self.indexed_matrix, kernel_type='Hex', radius=3, retina=True).type(torch.float),
            self.neighbours_indices_retina_hex_3
        )

    def test_square_to_hexagonal(self):
        torch.testing.assert_allclose(utils.square_to_hexagonal(self.square_image),
                                      self.square_image_hex)

    def test_square_to_hexagonal_index_matrix(self):
        torch.testing.assert_allclose(utils.square_to_hexagonal_index_matrix(self.square_image),
                                      self.square_to_hexagonal_index_matrix)

    def test_build_hexagonal_position(self):
        torch.testing.assert_allclose(utils.build_hexagonal_position(self.square_to_hexagonal_index_matrix),
                                      self.square_to_hexagonal_positions)

    def test_build_kernel(self):
        torch.testing.assert_allclose(utils.build_kernel('Hex'),
                                      self.hexagonal_kernel_1)
        torch.testing.assert_allclose(utils.build_kernel('Hex', dilation=2),
                                      self.hexagonal_kernel_1_2)
        torch.testing.assert_allclose(utils.build_kernel('Hex', radius=2),
                                      self.hexagonal_kernel_2)
        torch.testing.assert_allclose(utils.build_kernel('Square'),
                                      self.square_kernel_1)
        torch.testing.assert_allclose(utils.build_kernel('Square', dilation=2),
                                      self.square_kernel_1_2)
        torch.testing.assert_allclose(utils.build_kernel('Square', radius=0),
                                      self.square_kernel_0)


if __name__ == '__main__':
    unittest.main()
