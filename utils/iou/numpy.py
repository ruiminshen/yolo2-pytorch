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


def iou(yx_min1, yx_max1, yx_min2, yx_max2, min=None):
    """
    Calculates the IoU of two bounding boxes.
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min1: The top left coordinates (y, x) of the first bounding boxe.
    :param yx_max1: The bottom right coordinates (y, x) of the first bounding boxe.
    :param yx_min2: The top left coordinates (y, x) of the second bounding boxe.
    :param yx_max2: The bottom right coordinates (y, x) of the second bounding boxe.
    :return: The IoU.
    """
    assert np.all(yx_min1 <= yx_max1)
    assert np.all(yx_min2 <= yx_max2)
    if min is None:
        min = np.finfo(yx_min1.dtype).eps
    yx_min = np.maximum(yx_min1, yx_min2)
    yx_max = np.minimum(yx_max1, yx_max2)
    intersect_area = np.multiply.reduce(yx_max - yx_min)
    area1 = np.multiply.reduce(yx_max1 - yx_min1)
    area2 = np.multiply.reduce(yx_max2 - yx_min2)
    assert np.all(intersect_area >= 0)
    assert np.all(intersect_area <= area1)
    assert np.all(intersect_area <= area2)
    union_area = np.maximum(area1 + area2 - intersect_area, min)
    return intersect_area / union_area


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
    ymin1, xmin1 = yx_min1.T
    ymax1, xmax1 = yx_max1.T
    ymin2, xmin2 = yx_min2.T
    ymax2, xmax2 = yx_max2.T
    ymin1, xmin1, ymax1, xmax1, ymin2, xmin2, ymax2, xmax2 = (np.expand_dims(a, -1) for a in (ymin1, xmin1, ymax1, xmax1, ymin2, xmin2, ymax2, xmax2))
    max_ymin = np.maximum(ymin1, np.transpose(ymin2))
    min_ymax = np.minimum(ymax1, np.transpose(ymax2))
    height = np.maximum(0.0, min_ymax - max_ymin)
    max_xmin = np.maximum(xmin1, np.transpose(xmin2))
    min_xmax = np.minimum(xmax1, np.transpose(xmax2))
    width = np.maximum(0.0, min_xmax - max_xmin)
    return height * width


def iou_matrix(yx_min1, yx_max1, yx_min2, yx_max2, min=None):
    """
    Calculates the IoU of two lists of bounding boxes.
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min1: The top left coordinates (y, x) of the first list (size [N1, 2]) of bounding boxes.
    :param yx_max1: The bottom right coordinates (y, x) of the first list (size [N1, 2]) of bounding boxes.
    :param yx_min2: The top left coordinates (y, x) of the second list (size [N2, 2]) of bounding boxes.
    :param yx_max2: The bottom right coordinates (y, x) of the second list (size [N2, 2]) of bounding boxes.
    :return: The matrix (size [N1, N2]) of the IoU.
    """
    if min is None:
        min = np.finfo(yx_min1.dtype).eps
    assert np.all(yx_min1 <= yx_max1)
    assert np.all(yx_min2 <= yx_max2)
    intersect_area = intersection_area(yx_min1, yx_max1, yx_min2, yx_max2)
    area1 = np.expand_dims(np.multiply.reduce(yx_max1 - yx_min1, -1), 1)
    area2 = np.expand_dims(np.multiply.reduce(yx_max2 - yx_min2, -1), 0)
    assert np.all(intersect_area >= 0)
    assert np.all(intersect_area <= area1)
    assert np.all(intersect_area <= area2)
    union_area = np.maximum(area1 + area2 - intersect_area, min)
    return intersect_area / union_area


class TestIouMatrix(unittest.TestCase):
    def _test(self, bbox1, bbox2, ans, dtype=np.float32):
        bbox1, bbox2, ans = (np.array(a, dtype) for a in (bbox1, bbox2, ans))
        yx_min1, yx_max1 = np.split(bbox1, 2, -1)
        yx_min2, yx_max2 = np.split(bbox2, 2, -1)
        assert np.all(yx_min1 <= yx_max1)
        assert np.all(yx_min2 <= yx_max2)
        assert np.all(ans >= 0)
        matrix = iou_matrix(yx_min1, yx_max1, yx_min2, yx_max2)
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
