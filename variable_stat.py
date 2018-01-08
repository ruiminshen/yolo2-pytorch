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
import inspect
import operator

import yaml
import numpy as np
import torch
import humanize

import utils
import utils.train


def name(name, variable):
    return name


def size(name, variable):
    return 'x'.join(map(str, variable.size()))


def bytes(name, variable):
    return variable.numpy().nbytes


def natural_bytes(name, variable):
    return humanize.naturalsize(variable.numpy().nbytes)


def abs_mean(name, variable):
    return np.mean(np.abs(variable.numpy()))


def min_abs_mean(name, variable):
    return np.min([np.mean(np.abs(a)) for a in variable.numpy()])


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    model_dir = utils.get_model_dir(config)
    path, step, epoch = utils.train.load_model(model_dir)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['dnn']
    sig = inspect.signature(size)
    mapper = utils.load_functions(__file__)
    mapper = [(key, fn) for key, fn in mapper if inspect.signature(fn).parameters == sig.parameters]
    if not args.nohead:
        print('\t'.join(map(operator.itemgetter(0), mapper)))
    for name, variable in state_dict.items():
        row = (fn(name, variable) for key, fn in mapper)
        print('\t'.join(map(str, row)))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    parser.add_argument('--nohead', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
