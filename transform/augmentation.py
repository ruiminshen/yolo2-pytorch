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

import inspect
import random

import inflection
import numpy as np
import cv2

import transform


class Rotator(object):
    def __init__(self, y, x, height, width, angle):
        """
        A efficient tool to rotate multiple images in the same size.
        :author 申瑞珉 (Ruimin Shen)
        :param y: The y coordinate of rotation point.
        :param x: The x coordinate of rotation point.
        :param height: Image height.
        :param width: Image width.
        :param angle: Rotate angle.
        """
        self._mat = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        r = np.abs(self._mat[0, :2])
        _height, _width = np.inner(r, [height, width]), np.inner(r, [width, height])
        fix_y, fix_x = _height / 2 - y, _width / 2 - x
        self._mat[:, 2] += [fix_x, fix_y]
        self._size = int(_width), int(_height)

    def __call__(self, image, flags=cv2.INTER_LINEAR, fill=None):
        if fill is None:
            fill = np.random.rand(3) * 256
        return cv2.warpAffine(image, self._mat, self._size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    def _rotate_points(self, points):
        _points = np.pad(points, [(0, 0), (0, 1)], 'constant')
        _points[:, 2] = 1
        _points = np.dot(self._mat, _points.T)
        return _points.T.astype(points.dtype)

    def rotate_points(self, points):
        return self._rotate_points(points[:, ::-1])[:, ::-1]


def random_rotate(config, image, yx_min, yx_max):
    name = inspect.stack()[0][3]
    angle = random.uniform(*tuple(map(float, config.get('augmentation', name).split())))
    height, width = image.shape[:2]
    p1, p2 = np.copy(yx_min), np.copy(yx_max)
    p1[:, 0] = yx_max[:, 0]
    p2[:, 0] = yx_min[:, 0]
    points = np.concatenate([yx_min, yx_max, p1, p2], 0)
    rotator = Rotator(height / 2, width / 2, height, width, angle)
    image = rotator(image, fill=0)
    points = rotator.rotate_points(points)
    bbox_points = np.reshape(points, [4, -1, 2])
    yx_min = np.apply_along_axis(lambda points: np.min(points, 0), 0, bbox_points)
    yx_max = np.apply_along_axis(lambda points: np.max(points, 0), 0, bbox_points)
    return image, yx_min, yx_max


class RandomRotate(object):
    def __init__(self, config):
        self.config = config
        self.fn = eval(inflection.underscore(type(self).__name__))

    def __call__(self, data):
        data['image'], data['yx_min'], data['yx_max'] = self.fn(self.config, data['image'], data['yx_min'], data['yx_max'])
        return data


def flip_horizontally(image, yx_min, yx_max):
    assert len(image.shape) == 3
    image = cv2.flip(image, 1)
    width = image.shape[1]
    temp = width - yx_min[:, 1]
    yx_min[:, 1] = width - yx_max[:, 1]
    yx_max[:, 1] = temp
    return image, yx_min, yx_max


def random_flip_horizontally(config, image, yx_min, yx_max):
    name = inspect.stack()[0][3]
    if random.random() > config.getfloat('augmentation', name):
        return flip_horizontally(image, yx_min, yx_max)
    else:
        return image, yx_min, yx_max


class RandomFlipHorizontally(object):
    def __init__(self, config):
        self.config = config
        self.fn = eval(inflection.underscore(type(self).__name__))

    def __call__(self, data):
        data['image'], data['yx_min'], data['yx_max'] = self.fn(self.config, data['image'], data['yx_min'], data['yx_max'])
        return data


def get_transform(config, sequence):
    return transform.get_transform(config, sequence)
