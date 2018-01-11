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
import struct
import collections
import shutil
import hashlib
import yaml

import numpy as np
import torch
import humanize

import utils.train


def transpose_weight(weight, num_anchors):
    _, channels_in, ksize1, ksize2 = weight.size()
    weight = weight.view(num_anchors, -1, channels_in, ksize1, ksize2)
    x = weight[:, 0:1, :, :, :]
    y = weight[:, 1:2, :, :, :]
    w = weight[:, 2:3, :, :, :]
    h = weight[:, 3:4, :, :, :]
    iou = weight[:, 4:5, :, :, :]
    cls = weight[:, 5:, :, :, :]
    return torch.cat([iou, y, x, h, w, cls], 1).view(-1, channels_in, ksize1, ksize2)


def transpose_bias(bias, num_anchors):
    bias = bias.view([num_anchors, -1])
    x = bias[:, 0:1]
    y = bias[:, 1:2]
    w = bias[:, 2:3]
    h = bias[:, 3:4]
    iou = bias[:, 4:5]
    cls = bias[:, 5:]
    return torch.cat([iou, y, x, h, w, cls], 1).view(-1)


def group_state(state_dict):
    grouped_dict = collections.OrderedDict()
    for key, var in state_dict.items():
        layer, suffix1, suffix2 = key.rsplit('.', 2)
        suffix = suffix1 + '.' + suffix2
        if layer in grouped_dict:
            grouped_dict[layer][suffix] = var
        else:
            grouped_dict[layer] = {suffix: var}
    return grouped_dict


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    cache_dir = utils.get_cache_dir(config)
    model_dir = utils.get_model_dir(config)
    category = utils.get_category(config, cache_dir if os.path.exists(cache_dir) else None)
    anchors = utils.get_anchors(config)
    anchors = torch.from_numpy(anchors).contiguous()
    dnn = utils.parse_attr(config.get('model', 'dnn'))(config, anchors, len(category))
    dnn.eval()
    logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in dnn.state_dict().values())))
    state_dict = dnn.state_dict()
    grouped_dict = group_state(state_dict)
    try:
        layers = []
        with open(os.path.expanduser(os.path.expandvars(args.file)), 'rb') as f:
            major, minor, revision, seen = struct.unpack('4i', f.read(16))
            logging.info('major=%d, minor=%d, revision=%d, seen=%d' % (major, minor, revision, seen))
            total = 0
            filesize = os.fstat(f.fileno()).st_size
            for layer in grouped_dict:
                group = grouped_dict[layer]
                for suffix in ['conv.bias', 'bn.bias', 'bn.weight', 'bn.running_mean', 'bn.running_var', 'conv.weight']:
                    if suffix in group:
                        var = group[suffix]
                        size = var.size()
                        cnt = np.multiply.reduce(size)
                        total += cnt
                        key = layer + '.' + suffix
                        val = np.array(struct.unpack('%df' % cnt, f.read(cnt * 4)), np.float32)
                        val = np.reshape(val, size)
                        remaining = filesize - f.tell()
                        logging.info('%s.%s: %s=%f (%s), remaining=%d' % (layer, suffix, 'x'.join(list(map(str, size))), utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest(), remaining))
                        layers.append([key, torch.from_numpy(val)])
                logging.info('%d parameters assigned' % total)
        layers[-1][1] = transpose_weight(layers[-1][1], len(anchors))
        layers[-2][1] = transpose_bias(layers[-2][1], len(anchors))
    finally:
        if remaining > 0:
            logging.warning('%d bytes remaining' % remaining)
        state_dict.update(dict(layers))
        dnn.load_state_dict(state_dict)
        if args.delete:
            logging.warning('delete model directory: ' + model_dir)
            shutil.rmtree(model_dir, ignore_errors=True)
        saver = utils.train.Saver(model_dir, config.getint('save', 'keep'))
        saver(dict(dnn=state_dict), 0, 0)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Darknet .weights file')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
