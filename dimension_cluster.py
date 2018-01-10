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

import yaml
import numpy as np
import nltk.cluster.kmeans

import utils.data
import utils.iou.numpy


def distance(a, b):
    return 1 - utils.iou.numpy.iou(-a, a, -b, b)


def get_data(paths):
    dataset = utils.data.Dataset(utils.data.load_pickles(paths))
    return np.concatenate([(data['yx_max'] - data['yx_min']) / utils.image_size(data['path']) for data in dataset.dataset])


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    cache_dir = utils.get_cache_dir(config)
    paths = [os.path.join(cache_dir, phase + '.pkl') for phase in args.phase]
    data = get_data(paths)
    logging.info('num_examples=%d' % len(data))
    clusterer = nltk.cluster.kmeans.KMeansClusterer(args.num, distance, args.repeats)
    try:
        clusterer.cluster(data)
    except KeyboardInterrupt:
        logging.warning('interrupted')
    for m in clusterer.means():
        print('\t'.join(map(str, m)))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('num', type=int)
    parser.add_argument('-r', '--repeats', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
