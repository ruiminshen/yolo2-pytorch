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
import argparse
import configparser
import logging
import logging.config
import multiprocessing
import yaml

import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt

import utils.data
import utils.train
import utils.visualize
import transform.augmentation


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    cache_dir = utils.get_cache_dir(config)
    category = utils.get_category(config, cache_dir)
    draw_bbox = utils.visualize.DrawBBox(category)
    batch_size = args.rows * args.cols
    paths = [os.path.join(cache_dir, phase + '.pkl') for phase in args.phase]
    dataset = utils.data.Dataset(
        utils.data.load_pickles(paths),
        transform=transform.augmentation.get_transform(config, config.get('transform', 'augmentation').split()),
        shuffle=config.getboolean('data', 'shuffle'),
    )
    logging.info('num_examples=%d' % len(dataset))
    try:
        workers = config.getint('data', 'workers')
    except configparser.NoOptionError:
        workers = multiprocessing.cpu_count()
    collate_fn = utils.data.Collate(
        transform.parse_transform(config, config.get('transform', 'resize_train')),
        utils.train.load_sizes(config),
        maintain=config.getint('data', 'maintain'),
        transform_image=transform.get_transform(config, config.get('transform', 'image_train').split()),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    for data in loader:
        path, size, image, yx_min, yx_max, cls = (t.numpy() if hasattr(t, 'numpy') else t for t in (data[key] for key in 'path, size, image, yx_min, yx_max, cls'.split(', ')))
        fig, axes = plt.subplots(args.rows, args.cols)
        axes = axes.flat if batch_size > 1 else [axes]
        for ax, path, size, image, yx_min, yx_max, cls in zip(*[axes, path, size, image, yx_min, yx_max, cls]):
            logging.info(path + ': ' + 'x'.join(map(str, size)))
            size = yx_max - yx_min
            target = np.logical_and(*[np.squeeze(a, -1) > 0 for a in np.split(size, size.shape[-1], -1)])
            yx_min, yx_max, cls = (a[target] for a in (yx_min, yx_max, cls))
            image = draw_bbox(image, yx_min.astype(np.int), yx_max.astype(np.int), cls)
            ax.imshow(image)
            ax.set_title('%d objects' % np.sum(target))
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--rows', default=3, type=int)
    parser.add_argument('--cols', default=3, type=int)
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
