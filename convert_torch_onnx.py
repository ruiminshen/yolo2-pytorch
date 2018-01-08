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
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import torch.onnx
import humanize

import utils.train
import model


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    height, width = tuple(map(int, config.get('image', 'size').split()))
    cache_dir = utils.get_cache_dir(config)
    model_dir = utils.get_model_dir(config)
    category = utils.get_category(config, cache_dir if os.path.exists(cache_dir) else None)
    anchors = utils.get_anchors(config)
    anchors = torch.from_numpy(anchors).contiguous()
    dnn = utils.parse_attr(config.get('model', 'dnn'))(config, anchors, len(category))
    inference = model.Inference(config, dnn, anchors)
    inference.eval()
    logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in inference.state_dict().values())))
    path, step, epoch = utils.train.load_model(model_dir)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    dnn.load_state_dict(checkpoint['dnn'])
    image = torch.autograd.Variable(torch.randn(args.batch_size, 3, height, width))
    path = model_dir + '.onnx'
    logging.info('save ' + path)
    torch.onnx.export(dnn, image, path, export_params=True, verbose=args.verbose) # PyTorch's bug


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
