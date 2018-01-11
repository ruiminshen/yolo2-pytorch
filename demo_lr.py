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
import logging.config
import os
import yaml

import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data

import model
import utils.data
import utils.postprocess
import utils.train
import utils.visualize


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    category = utils.get_category(config)
    anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
    dnn = utils.parse_attr(config.get('model', 'dnn'))(config, anchors, len(category))
    inference = model.Inference(config, dnn, anchors)
    inference.train()
    optimizer = eval(config.get('train', 'optimizer'))(filter(lambda p: p.requires_grad, inference.parameters()), args.learning_rate)
    scheduler = eval(config.get('train', 'scheduler'))(optimizer)
    for epoch in range(args.epoch):
        scheduler.step(epoch)
        lr = scheduler.get_lr()
        print('\t'.join(map(str, [epoch] + lr)))



def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int)
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-o', '--optimizer', default='adam')
    parser.add_argument('-lr', '--learning_rate', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
