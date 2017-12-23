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

import random

import numpy as np
import torchvision
import inflection
import skimage.exposure
import cv2


class BGR2RGB(object):
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class BGR2HSV(object):
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


class HSV2RGB(object):
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


class RandomBlur(object):
    def __init__(self, config):
        name = inflection.underscore(type(self).__name__)
        self.adjust = tuple(map(int, config.get('augmentation', name).split()))

    def __call__(self, image):
        adjust = tuple(random.randint(1, adjust) for adjust in self.adjust)
        return cv2.blur(image, adjust)


class RandomHue(object):
    def __init__(self, config):
        name = inflection.underscore(type(self).__name__)
        self.adjust = tuple(map(int, config.get('augmentation', name).split()))

    def __call__(self, hsv):
        h, s, v = cv2.split(hsv)
        adjust = random.randint(*self.adjust)
        h = h.astype(np.int) + adjust
        h = np.clip(h, 0, 179).astype(hsv.dtype)
        return cv2.merge((h, s, v))


class RandomSaturation(object):
    def __init__(self, config):
        name = inflection.underscore(type(self).__name__)
        self.adjust = tuple(map(float, config.get('augmentation', name).split()))

    def __call__(self, hsv):
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(*self.adjust)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        return cv2.merge((h, s, v))


class RandomBrightness(object):
    def __init__(self, config):
        name = inflection.underscore(type(self).__name__)
        self.adjust = tuple(map(float, config.get('augmentation', name).split()))

    def __call__(self, hsv):
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(*self.adjust)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        return cv2.merge((h, s, v))


class RandomGamma(object):
    def __init__(self, config):
        name = inflection.underscore(type(self).__name__)
        self.adjust = tuple(map(float, config.get('augmentation', name).split()))

    def __call__(self, image):
        adjust = random.uniform(*self.adjust)
        return skimage.exposure.adjust_gamma(image, adjust)


class Normalize(torchvision.transforms.Normalize):
    def __init__(self, config):
        torchvision.transforms.Normalize.__init__(self, (0.5, 0.5, 0.5), (1, 1, 1))
