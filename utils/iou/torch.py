"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest

import numpy as np
import torch


def intersection_area(yx_min1, yx_max1, yx_min2, yx_max2):
    """
    Calculates the intersection area of two lists of bounding boxes.
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min1: The top left coordinates (y, x) of the first list (size [N1, 2]) of bounding boxes.
    :param yx_max1: The bottom right coordinates (y, x) of the first list (size [N1, 2]) of bounding boxes.
    :param yx_min2: The top left coordinates (y, x) of the second list (size [N2, 2]) of bounding boxes.
    :param yx_max2: The bottom right coordinates (y, x) of the second list (size [N2, 2]) of bounding boxes.
    :return: The matrix (size [N1, N2]) of the intersection area.
    """
    ymin1, xmin1 = torch.split(yx_min1, 1, -1)
    ymax1, xmax1 = torch.split(yx_max1, 1, -1)
    ymin2, xmin2 = torch.split(yx_min2, 1, -1)
    ymax2, xmax2 = torch.split(yx_max2, 1, -1)
    max_ymin = torch.max(ymin1.repeat(1, ymin2.size(0)), torch.transpose(ymin2, 0, 1).repeat(ymin1.size(0), 1)) # PyTorch's bug
    min_ymax = torch.min(ymax1.repeat(1, ymax2.size(0)), torch.transpose(ymax2, 0, 1).repeat(ymax1.size(0), 1)) # PyTorch's bug
    height = torch.clamp(min_ymax - max_ymin, min=0)
    max_xmin = torch.max(xmin1.repeat(1, xmin2.size(0)), torch.transpose(xmin2, 0, 1).repeat(xmin1.size(0), 1)) # PyTorch's bug
    min_xmax = torch.min(xmax1.repeat(1, xmax2.size(0)), torch.transpose(xmax2, 0, 1).repeat(xmax1.size(0), 1)) # PyTorch's bug
    width = torch.clamp(min_xmax - max_xmin, min=0)
    return height * width


def iou_matrix(yx_min1, yx_max1, yx_min2, yx_max2, min=float(np.finfo(np.float32).eps)):
    """
    Calculates the IoU of two lists of bounding boxes.
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min1: The top left coordinates (y, x) of the first list (size [N1, 2]) of bounding boxes.
    :param yx_max1: The bottom right coordinates (y, x) of the first list (size [N1, 2]) of bounding boxes.
    :param yx_min2: The top left coordinates (y, x) of the second list (size [N2, 2]) of bounding boxes.
    :param yx_max2: The bottom right coordinates (y, x) of the second list (size [N2, 2]) of bounding boxes.
    :return: The matrix (size [N1, N2]) of the IoU.
    """
    intersect_area = intersection_area(yx_min1, yx_max1, yx_min2, yx_max2)
    area1 = torch.prod(yx_max1 - yx_min1, -1).unsqueeze(-1)
    area2 = torch.prod(yx_max2 - yx_min2, -1).unsqueeze(-2)
    union_area = torch.clamp(area1 + area2 - intersect_area, min=min)
    return intersect_area / union_area


class TestIouMatrix(unittest.TestCase):
    def _test(self, bbox1, bbox2, ans, dtype=np.float32):
        bbox1, bbox2, ans = (np.array(a, dtype) for a in (bbox1, bbox2, ans))
        yx_min1, yx_max1 = np.split(bbox1, 2, -1)
        yx_min2, yx_max2 = np.split(bbox2, 2, -1)
        assert np.all(yx_min1 <= yx_max1)
        assert np.all(yx_min2 <= yx_max2)
        assert np.all(ans >= 0)
        yx_min1, yx_max1 = torch.autograd.Variable(torch.from_numpy(yx_min1)), torch.autograd.Variable(torch.from_numpy(yx_max1))
        yx_min2, yx_max2 = torch.autograd.Variable(torch.from_numpy(yx_min2)), torch.autograd.Variable(torch.from_numpy(yx_max2))
        if torch.cuda.is_available():
            yx_min1, yx_max1, yx_min2, yx_max2 = (v.cuda() for v in (yx_min1, yx_max1, yx_min2, yx_max2))
        matrix = iou_matrix(yx_min1, yx_max1, yx_min2, yx_max2).data.cpu().numpy()
        np.testing.assert_almost_equal(matrix, ans)

    def test0(self):
        bbox1 = [
            (1, 1, 2, 2),
        ]
        bbox2 = [
            (0, 0, 1, 1),
            (0, 1, 1, 2),
            (0, 2, 1, 3),
            (1, 0, 2, 1),
            (2, 0, 3, 1),
            (1, 2, 2, 3),
            (2, 1, 3, 2),
            (2, 2, 3, 3),
        ]
        ans = [
            [0] * len(bbox2),
        ]
        self._test(bbox1, bbox2, ans)

    def test1(self):
        bbox1 = [
            (1, 1, 3, 3),
            (0, 0, 4, 4),
        ]
        bbox2 = [
            (0, 0, 2, 2),
            (2, 0, 4, 2),
            (0, 2, 2, 4),
            (2, 2, 4, 4),
        ]
        ans = [
            [1 / (4 + 4 - 1)] * len(bbox2),
            [4 / 16] * len(bbox2),
        ]
        self._test(bbox1, bbox2, ans)


def batch_intersection_area(yx_min1, yx_max1, yx_min2, yx_max2):
    """
    Calculates the intersection area of two lists of bounding boxes for N independent batches.
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min1: The top left coordinates (y, x) of the first lists (size [N, N1, 2]) of bounding boxes.
    :param yx_max1: The bottom right coordinates (y, x) of the first lists (size [N, N1, 2]) of bounding boxes.
    :param yx_min2: The top left coordinates (y, x) of the second lists (size [N, N2, 2]) of bounding boxes.
    :param yx_max2: The bottom right coordinates (y, x) of the second lists (size [N, N2, 2]) of bounding boxes.
    :return: The matrics (size [N, N1, N2]) of the intersection area.
    """
    ymin1, xmin1 = torch.split(yx_min1, 1, -1)
    ymax1, xmax1 = torch.split(yx_max1, 1, -1)
    ymin2, xmin2 = torch.split(yx_min2, 1, -1)
    ymax2, xmax2 = torch.split(yx_max2, 1, -1)
    max_ymin = torch.max(ymin1.repeat(1, 1, ymin2.size(1)), torch.transpose(ymin2, 1, 2).repeat(1, ymin1.size(1), 1)) # PyTorch's bug
    min_ymax = torch.min(ymax1.repeat(1, 1, ymax2.size(1)), torch.transpose(ymax2, 1, 2).repeat(1, ymax1.size(1), 1)) # PyTorch's bug
    height = torch.clamp(min_ymax - max_ymin, min=0)
    max_xmin = torch.max(xmin1.repeat(1, 1, xmin2.size(1)), torch.transpose(xmin2, 1, 2).repeat(1, xmin1.size(1), 1)) # PyTorch's bug
    min_xmax = torch.min(xmax1.repeat(1, 1, xmax2.size(1)), torch.transpose(xmax2, 1, 2).repeat(1, xmax1.size(1), 1)) # PyTorch's bug
    width = torch.clamp(min_xmax - max_xmin, min=0)
    return height * width


def batch_iou_matrix(yx_min1, yx_max1, yx_min2, yx_max2, min=float(np.finfo(np.float32).eps)):
    """
    Calculates the IoU of two lists of bounding boxes for N independent batches.
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min1: The top left coordinates (y, x) of the first lists (size [N, N1, 2]) of bounding boxes.
    :param yx_max1: The bottom right coordinates (y, x) of the first lists (size [N, N1, 2]) of bounding boxes.
    :param yx_min2: The top left coordinates (y, x) of the second lists (size [N, N2, 2]) of bounding boxes.
    :param yx_max2: The bottom right coordinates (y, x) of the second lists (size [N, N2, 2]) of bounding boxes.
    :return: The matrics (size [N, N1, N2]) of the IoU.
    """
    intersect_area = batch_intersection_area(yx_min1, yx_max1, yx_min2, yx_max2)
    area1 = torch.prod(yx_max1 - yx_min1, -1).unsqueeze(-1)
    area2 = torch.prod(yx_max2 - yx_min2, -1).unsqueeze(-2)
    union_area = torch.clamp(area1 + area2 - intersect_area, min=min)
    return intersect_area / union_area


class TestBatchIouMatrix(unittest.TestCase):
    def _test(self, bbox1, bbox2, ans, batch_size=2, dtype=np.float32):
        bbox1, bbox2, ans = (np.expand_dims(np.array(a, dtype), 0) for a in (bbox1, bbox2, ans))
        if batch_size > 1:
            bbox1, bbox2, ans = (np.tile(a, (batch_size, 1, 1)) for a in (bbox1, bbox2, ans))
            for b in range(batch_size):
                indices1 = np.random.permutation(bbox1.shape[1])
                indices2 = np.random.permutation(bbox2.shape[1])
                bbox1[b] = bbox1[b][indices1]
                bbox2[b] = bbox2[b][indices2]
                ans[b] = ans[b][indices1][:, indices2]
        yx_min1, yx_max1 = np.split(bbox1, 2, -1)
        yx_min2, yx_max2 = np.split(bbox2, 2, -1)
        assert np.all(yx_min1 <= yx_max1)
        assert np.all(yx_min2 <= yx_max2)
        assert np.all(ans >= 0)
        yx_min1, yx_max1 = torch.autograd.Variable(torch.from_numpy(yx_min1)), torch.autograd.Variable(torch.from_numpy(yx_max1))
        yx_min2, yx_max2 = torch.autograd.Variable(torch.from_numpy(yx_min2)), torch.autograd.Variable(torch.from_numpy(yx_max2))
        if torch.cuda.is_available():
            yx_min1, yx_max1, yx_min2, yx_max2 = (v.cuda() for v in (yx_min1, yx_max1, yx_min2, yx_max2))
        matrix = batch_iou_matrix(yx_min1, yx_max1, yx_min2, yx_max2).data.cpu().numpy()
        np.testing.assert_almost_equal(matrix, ans)

    def test0(self):
        bbox1 = [
            (1, 1, 2, 2),
        ]
        bbox2 = [
            (0, 0, 1, 1),
            (0, 1, 1, 2),
            (0, 2, 1, 3),
            (1, 0, 2, 1),
            (2, 0, 3, 1),
            (1, 2, 2, 3),
            (2, 1, 3, 2),
            (2, 2, 3, 3),
        ]
        ans = [
            [0] * len(bbox2),
        ]
        self._test(bbox1, bbox2, ans)

    def test1(self):
        bbox1 = [
            (1, 1, 3, 3),
            (0, 0, 4, 4),
        ]
        bbox2 = [
            (0, 0, 2, 2),
            (2, 0, 4, 2),
            (0, 2, 2, 4),
            (2, 2, 4, 4),
        ]
        ans = [
            [1 / (4 + 4 - 1)] * len(bbox2),
            [4 / 16] * len(bbox2),
        ]
        self._test(bbox1, bbox2, ans)


def batch_iou_pair(yx_min1, yx_max1, yx_min2, yx_max2, min=float(np.finfo(np.float32).eps)):
    """
    Pairwisely calculates the IoU of two lists (at the same size M) of bounding boxes for N independent batches.
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min1: The top left coordinates (y, x) of the first lists (size [N, M, 2]) of bounding boxes.
    :param yx_max1: The bottom right coordinates (y, x) of the first lists (size [N, M, 2]) of bounding boxes.
    :param yx_min2: The top left coordinates (y, x) of the second lists (size [N, M, 2]) of bounding boxes.
    :param yx_max2: The bottom right coordinates (y, x) of the second lists (size [N, M, 2]) of bounding boxes.
    :return: The lists (size [N, M]) of the IoU.
    """
    yx_min = torch.max(yx_min1, yx_min2)
    yx_max = torch.min(yx_max1, yx_max2)
    size = torch.clamp(yx_max - yx_min, min=0)
    intersect_area = torch.prod(size, -1)
    area1 = torch.prod(yx_max1 - yx_min1, -1)
    area2 = torch.prod(yx_max2 - yx_min2, -1)
    union_area = torch.clamp(area1 + area2 - intersect_area, min=min)
    return intersect_area / union_area


class TestBatchIouPair(unittest.TestCase):
    def _test(self, bbox1, bbox2, ans, dtype=np.float32):
        bbox1, bbox2, ans = (np.array(a, dtype) for a in (bbox1, bbox2, ans))
        batch_size = bbox1.shape[0]
        cells = bbox2.shape[0]
        bbox1 = np.tile(np.reshape(bbox1, [-1, 1, 4]), [1, cells, 1])
        bbox2 = np.tile(np.reshape(bbox2, [1, -1, 4]), [batch_size, 1, 1])
        yx_min1, yx_max1 = np.split(bbox1, 2, -1)
        yx_min2, yx_max2 = np.split(bbox2, 2, -1)
        assert np.all(yx_min1 <= yx_max1)
        assert np.all(yx_min2 <= yx_max2)
        assert np.all(ans >= 0)
        yx_min1, yx_max1 = torch.autograd.Variable(torch.from_numpy(yx_min1)), torch.autograd.Variable(torch.from_numpy(yx_max1))
        yx_min2, yx_max2 = torch.autograd.Variable(torch.from_numpy(yx_min2)), torch.autograd.Variable(torch.from_numpy(yx_max2))
        if torch.cuda.is_available():
            yx_min1, yx_max1, yx_min2, yx_max2 = (v.cuda() for v in (yx_min1, yx_max1, yx_min2, yx_max2))
        iou = batch_iou_pair(yx_min1, yx_max1, yx_min2, yx_max2).data.cpu().numpy()
        np.testing.assert_almost_equal(iou, ans)

    def test0(self):
        bbox1 = [
            (1, 1, 2, 2),
        ]
        bbox2 = [
            (0, 0, 1, 1),
            (0, 1, 1, 2),
            (0, 2, 1, 3),
            (1, 0, 2, 1),
            (2, 0, 3, 1),
            (1, 2, 2, 3),
            (2, 1, 3, 2),
            (2, 2, 3, 3),
        ]
        ans = [
            [0] * len(bbox2),
        ]
        self._test(bbox1, bbox2, ans)

    def test1(self):
        bbox1 = [
            (1, 1, 3, 3),
            (0, 0, 4, 4),
        ]
        bbox2 = [
            (0, 0, 2, 2),
            (2, 0, 4, 2),
            (0, 2, 2, 4),
            (2, 2, 4, 4),
        ]
        ans = [
            [1 / (4 + 4 - 1)] * len(bbox2),
            [4 / 16] * len(bbox2),
        ]
        self._test(bbox1, bbox2, ans)


if __name__ == '__main__':
    unittest.main()
