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
import logging
import configparser

import numpy as np
import tqdm
import xml.etree.ElementTree
import cv2

import utils.cache


def load_annotation(path, category_index):
    tree = xml.etree.ElementTree.parse(path)
    yx_min = []
    yx_max = []
    cls = []
    difficult = []
    for obj in tree.findall('object'):
        try:
            cls.append(category_index[obj.find('name').text])
        except KeyError:
            continue
        bbox = obj.find('bndbox')
        ymin = float(bbox.find('ymin').text) - 1
        xmin = float(bbox.find('xmin').text) - 1
        ymax = float(bbox.find('ymax').text) - 1
        xmax = float(bbox.find('xmax').text) - 1
        assert ymin < ymax
        assert xmin < xmax
        yx_min.append((ymin, xmin))
        yx_max.append((ymax, xmax))
        difficult.append(int(obj.find('difficult').text))
    size = tree.find('size')
    return tree.find('filename').text, (int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)), yx_min, yx_max, cls, difficult


def load_root():
    with open(os.path.splitext(__file__)[0] + '.txt', 'r') as f:
        return [line.rstrip() for line in f]


def cache(config, path, category_index, root=load_root()):
    phase = os.path.splitext(os.path.basename(path))[0]
    data = []
    for root in root:
        logging.info('loading ' + root)
        root = os.path.expanduser(os.path.expandvars(root))
        path = os.path.join(root, 'ImageSets', 'Main', phase) + '.txt'
        if not os.path.exists(path):
            logging.warning(path + ' not exists')
            continue
        with open(path, 'r') as f:
            filenames = [line.strip() for line in f]
        for filename in tqdm.tqdm(filenames):
            filename, size, yx_min, yx_max, cls, difficult = load_annotation(os.path.join(root, 'Annotations', filename + '.xml'), category_index)
            if len(cls) <= 0:
                continue
            path = os.path.join(root, 'JPEGImages', filename)
            yx_min = np.array(yx_min, dtype=np.float32)
            yx_max = np.array(yx_max, dtype=np.float32)
            cls = np.array(cls, dtype=np.int)
            difficult = np.array(difficult, dtype=np.uint8)
            assert len(yx_min) == len(cls)
            assert yx_min.shape == yx_max.shape
            assert len(yx_min.shape) == 2 and yx_min.shape[-1] == 2
            try:
                if config.getboolean('cache', 'verify'):
                    try:
                        image = cv2.imread(path)
                        assert image is not None
                        assert image.shape[:2] == size[:2]
                        utils.cache.verify_coords(yx_min, yx_max, size[:2])
                    except AssertionError as e:
                        logging.error(path + ': ' + str(e))
                        continue
            except configparser.NoOptionError:
                pass
            data.append(dict(path=path, yx_min=yx_min, yx_max=yx_max, cls=cls, difficult=difficult))
        logging.info('%d of %d images are saved' % (len(data), len(filenames)))
    return data
