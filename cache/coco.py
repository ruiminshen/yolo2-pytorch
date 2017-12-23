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
import pandas as pd
import tqdm
import pycocotools.coco
import cv2

import utils.cache


def cache(config, path, category_index):
    phase = os.path.splitext(os.path.basename(path))[0]
    data = []
    for i, row in pd.read_csv(os.path.splitext(__file__)[0] + '.tsv', sep='\t').iterrows():
        logging.info('loading data %d (%s)' % (i, ', '.join([k + '=' + str(v) for k, v in row.items()])))
        root = os.path.expanduser(os.path.expandvars(row['root']))
        year = str(row['year'])
        suffix = phase + year
        path = os.path.join(root, 'annotations', 'instances_%s.json' % suffix)
        if not os.path.exists(path):
            logging.warning(path + ' not exists')
            continue
        coco = pycocotools.coco.COCO(path)
        catIds = coco.getCatIds(catNms=list(category_index.keys()))
        cats = coco.loadCats(catIds)
        id_index = dict((cat['id'], category_index[cat['name']]) for cat in cats)
        imgIds = coco.getImgIds()
        path = os.path.join(root, suffix)
        imgs = coco.loadImgs(imgIds)
        _imgs = list(filter(lambda img: os.path.exists(os.path.join(path, img['file_name'])), imgs))
        if len(imgs) > len(_imgs):
            logging.warning('%d of %d images not exists' % (len(imgs) - len(_imgs), len(imgs)))
        for img in tqdm.tqdm(_imgs):
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            if len(anns) <= 0:
                continue
            path = os.path.join(path, img['file_name'])
            width, height = img['width'], img['height']
            bbox = np.array([ann['bbox'] for ann in anns], dtype=np.float32)
            yx_min = bbox[:, 1::-1]
            hw = bbox[:, -1:1:-1]
            yx_max = yx_min + hw
            cls = np.array([id_index[ann['category_id']] for ann in anns], dtype=np.int)
            difficult = np.zeros(cls.shape, dtype=np.uint8)
            try:
                if config.getboolean('cache', 'verify'):
                    size = (height, width)
                    image = cv2.imread(path)
                    assert image is not None
                    assert image.shape[:2] == size[:2]
                    utils.cache.verify_coords(yx_min, yx_max, size[:2])
            except configparser.NoOptionError:
                pass
            assert len(yx_min) == len(cls)
            assert yx_min.shape == yx_max.shape
            assert len(yx_min.shape) == 2 and yx_min.shape[-1] == 2
            data.append(dict(path=path, yx_min=yx_min, yx_max=yx_max, cls=cls, difficult=difficult))
        logging.warning('%d of %d images are saved' % (len(data), len(_imgs)))
    return data
