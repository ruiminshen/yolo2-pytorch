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
import hashlib
import yaml

import torch
import torch.autograd
import cv2

import utils
import utils.train
import model
import transform


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    torch.manual_seed(args.seed)
    cache_dir = utils.get_cache_dir(config)
    model_dir = utils.get_model_dir(config)
    category = utils.get_category(config, cache_dir if os.path.exists(cache_dir) else None)
    anchors = utils.get_anchors(config)
    anchors = torch.from_numpy(anchors).contiguous()
    path, step, epoch = utils.train.load_model(model_dir)
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    dnn = utils.parse_attr(config.get('model', 'dnn'))(model.ConfigChannels(config, state_dict), anchors, len(category))
    dnn.load_state_dict(state_dict)
    height, width = tuple(map(int, config.get('image', 'size').split()))
    tensor = torch.randn(1, 3, height, width)
    # Checksum
    for key, var in dnn.state_dict().items():
        a = var.cpu().numpy()
        print('\t'.join(map(str, [key, a.shape, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()])))
    output = dnn(torch.autograd.Variable(tensor, volatile=True)).data
    for key, a in [
        ('tensor', tensor.cpu().numpy()),
        ('output', output.cpu().numpy()),
    ]:
        print('\t'.join(map(str, [key, a.shape, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()])))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    parser.add_argument('-s', '--seed', default=0, type=int, help='a seed to create a random image tensor')
    return parser.parse_args()


if __name__ == '__main__':
    main()
