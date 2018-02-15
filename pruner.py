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
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import humanize

import model
import utils
import utils.train
import utils.walk


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    model_dir = utils.get_model_dir(config)
    category = utils.get_category(config)
    anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
    path, step, epoch = utils.train.load_model(model_dir)
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    _model = utils.parse_attr(config.get('model', 'dnn'))
    dnn = _model(model.ConfigChannels(config, state_dict), anchors, len(category))
    logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in dnn.state_dict().values())))
    dnn.load_state_dict(state_dict)
    height, width = tuple(map(int, config.get('image', 'size').split()))
    image = torch.autograd.Variable(torch.randn(args.batch_size, 3, height, width))
    output = dnn(image)
    state_dict = dnn.state_dict()
    d = utils.dense(state_dict[args.name])
    keep = torch.LongTensor(np.argsort(d)[int(len(d) * args.remove):])
    closure = utils.walk.Closure(
        args.name, state_dict, dnn,
        lambda name, var: var[keep],
        lambda name, var, mapper: var[mapper(keep, len(d))],
        debug=args.debug,
    )
    closure(output.grad_fn)
    if args.debug:
        path = closure.dot.view('%s.%s.gv' % (os.path.basename(model_dir), os.path.basename(os.path.splitext(__file__)[0])), os.path.dirname(model_dir))
        logging.info(path)
    assert len(keep) == len(state_dict[args.name])
    dnn = _model(model.ConfigChannels(config, state_dict), anchors, len(category))
    dnn.load_state_dict(state_dict)
    dnn(image)
    if not args.debug:
        torch.save(state_dict, path)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('remove', type=float)
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()