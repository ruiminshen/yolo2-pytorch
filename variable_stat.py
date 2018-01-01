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

import argparse
import configparser
import logging
import inspect
import operator

import numpy as np
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


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    if args.level:
        logging.getLogger().setLevel(args.level.upper())
    model_dir = utils.get_model_dir(config)
    checkpoint, step, epoch = utils.train.load_model(model_dir)
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
    parser.add_argument('--level', default='info', help='logging level')
    parser.add_argument('--nohead', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main()
