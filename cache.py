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
import importlib
import pickle
import random
import shutil

import utils


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    cache_dir = utils.get_cache_dir(config)
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copyfile(os.path.expanduser(os.path.expandvars(config.get('cache', 'category'))), os.path.join(cache_dir, 'category'))
    category = utils.get_category(config)
    category_index = dict([(name, i) for i, name in enumerate(category)])
    datasets = config.get('cache', 'datasets').split()
    for phase in args.phase:
        path = os.path.join(cache_dir, phase) + '.pkl'
        logging.info('save cache file: ' + path)
        data = []
        for dataset in datasets:
            logging.info('load %s dataset' % dataset)
            module, func = dataset.rsplit('.', 1)
            module = importlib.import_module(module)
            func = getattr(module, func)
            data += func(config, path, category_index)
        if config.getboolean('cache', 'shuffle'):
            random.shuffle(data)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    logging.info('%s data are saved into %s' % (str(args.phase), cache_dir))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()

if __name__ == '__main__':
    main()
