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

import os
import pickle
import random
import copy

import cv2
import numpy as np
import sklearn.preprocessing
import torch.utils.data

import transform.resize.label


def padding_labels(data, dim, labels='yx_min, yx_max, cls, difficult'.split(', ')):
    """
    Padding labels into the same dimension (to form a batch).
    :author 申瑞珉 (Ruimin Shen)
    :param data: A dict contains the labels to be padded.
    :param dim: The target dimension.
    :param labels: The list of label names.
    :return: The padded label dict.
    """
    pad = dim - len(data[labels[0]])
    for key in labels:
        label = data[key]
        data[key] = np.pad(label, [(0, pad)] + [(0, 0)] * (len(label.shape) - 1), 'constant')
    return data


def load_pickles(paths):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data += pickle.load(f)
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=lambda data: data, one_hot=None, shuffle=False, dir=None):
        """
        Load the cached data (.pkl) into memory.
        :author 申瑞珉 (Ruimin Shen)
        :param data: A list contains the data samples (dict).
        :param transform: A function transforms (usually performs a sequence of data augmentation operations) the labels in a dict.
        :param one_hot: If a int value (total number of classes) is given, the class label (key "cls") will be generated in a one-hot format.
        :param shuffle: Shuffle the loaded dataset.
        :param dir: The directory to store the exception data.
        """
        self.data = data
        if shuffle:
            random.shuffle(self.data)
        self.transform = transform
        self.one_hot = None if one_hot is None else sklearn.preprocessing.OneHotEncoder(one_hot, dtype=np.float32)
        self.dir = dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        try:
            image = cv2.imread(data['path'])
            data['image'] = image
            data['size'] = np.array(image.shape[:2])
            data = self.transform(data)
            if self.one_hot is not None:
                data['cls'] = self.one_hot.fit_transform(np.expand_dims(data['cls'], -1)).todense()
        except:
            if self.dir is not None:
                os.makedirs(self.dir, exist_ok=True)
                name = self.__module__ + '.' + type(self).__name__
                with open(os.path.join(self.dir, name + '.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            raise
        return data


class Collate(object):
    def __init__(self, sizes, maintain=1, resize=transform.resize.label.naive, transform_image=lambda image: image, transform_tensor=None, dir=None):
        """
        Unify multiple data samples (e.g., resize images into the same size, and padding bounding box labels into the same number) to form a batch.
        :author 申瑞珉 (Ruimin Shen)
        :param sizes: The image sizes to be randomly choosed.
        :param maintain: How many times a size to be maintained.
        :param resize: A function to resize the image and labels.
        :param transform_image: A function to transform the resized image.
        :param transform_tensor: A function to standardize a image into a tensor.
        :param dir: The directory to store the exception data.
        """
        self.resize = resize
        self.sizes = sizes
        assert maintain > 0
        self.maintain = maintain
        self._maintain = maintain
        self.transform_image = transform_image
        self.transform_tensor = transform_tensor
        self.dir = dir

    def __call__(self, batch):
        height, width = self.next_size()
        dim = max(len(data['cls']) for data in batch)
        _batch = []
        for data in batch:
            try:
                data = self.resize(data, height, width)
                data['image'] = self.transform_image(data['image'])
                data = padding_labels(data, dim)
                if self.transform_tensor is not None:
                    data['tensor'] = self.transform_tensor(data['image'])
                _batch.append(data)
            except:
                if self.dir is not None:
                    os.makedirs(self.dir, exist_ok=True)
                    name = self.__module__ + '.' + type(self).__name__
                    with open(os.path.join(self.dir, name + '.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                raise
        return torch.utils.data.dataloader.default_collate(_batch)

    def next_size(self):
        if self._maintain < self.maintain:
            self._maintain += 1
        else:
            self.size = random.choice(self.sizes)
            self._maintain = 0
        return self.size
