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
import inspect
import inflection
import yaml

import numpy as np
import torch
import humanize
import xlsxwriter

import utils
import utils.train
import utils.channel


class Name(object):
    def __call__(self, name, variable):
        return name


class Size(object):
    def __call__(self, name, variable):
        return 'x'.join(map(str, variable.size()))


class Bytes(object):
    def __call__(self, name, variable):
        return variable.numpy().nbytes

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class BytesNatural(object):
    def __call__(self, name, variable):
        return humanize.naturalsize(variable.numpy().nbytes)


class MeanDense(object):
    def __call__(self, name, variable):
        return np.mean(utils.channel.dense(variable))

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class Rank(object):
    def __call__(self, name, variable):
        return len(variable.size())

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


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
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    mapper = [(inflection.underscore(name), member()) for name, member in inspect.getmembers(importlib.machinery.SourceFileLoader('', __file__).load_module()) if inspect.isclass(member)]
    path = os.path.join(model_dir, os.path.basename(os.path.splitext(__file__)[0])) + '.xlsx'
    with xlsxwriter.Workbook(path, {'strings_to_urls': False, 'nan_inf_to_errors': True}) as workbook:
        worksheet = workbook.add_worksheet(args.worksheet)
        for j, (key, m) in enumerate(mapper):
            worksheet.write(0, j, key)
            for i, (name, variable) in enumerate(state_dict.items()):
                value = m(name, variable)
                worksheet.write(1 + i, j, value)
            if hasattr(m, 'format'):
                m.format(workbook, worksheet, i, j)
        worksheet.autofilter(0, 0, i, len(mapper) - 1)
        worksheet.freeze_panes(1, 0)
    logging.info(path)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    parser.add_argument('--worksheet', default='sheet')
    parser.add_argument('--nohead', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
