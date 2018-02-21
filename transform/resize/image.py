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

import inflection
import numpy as np
import cv2


def rescale(image, height, width):
    return cv2.resize(image, (width, height))


class Rescale(object):
    def __init__(self):
        self.fn = eval(inflection.underscore(type(self).__name__))

    def __call__(self, image, height, width):
        return self.fn(image, height, width)


def fixed(image, height, width):
    _height, _width, _ = image.shape
    if _height / _width > height / width:
        scale = height / _height
    else:
        scale = width / _width
    m = np.eye(2, 3)
    m[0, 0] = scale
    m[1, 1] = scale
    flags = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.warpAffine(image, m, (width, height), flags=flags)


class Fixed(object):
    def __init__(self):
        self.fn = eval(inflection.underscore(type(self).__name__))

    def __call__(self, image, height, width):
        return self.fn(image, height, width)


class Resize(object):
    def __init__(self, config):
        self.fn = eval(config.get('data', inflection.underscore(type(self).__name__)))

    def __call__(self, image, height, width):
        return self.fn(image, height, width)
