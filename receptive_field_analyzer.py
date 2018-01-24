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
import multiprocessing
import yaml

import numpy as np
import scipy.misc
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import tqdm
import humanize

import utils.data
import utils.iou.torch
import utils.postprocess
import utils.train
import utils.visualize


class Dataset(torch.utils.data.Dataset):
    def __init__(self, height, width):
        self.points = np.array([(i, j) for i in range(height) for j in range(width)])

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]


class Analyzer(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.model_dir = utils.get_model_dir(config)
        self.category = utils.get_category(config)
        self.anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
        self.dnn = utils.parse_attr(config.get('model', 'dnn'))(config, self.anchors, len(self.category))
        self.dnn.eval()
        logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in self.dnn.state_dict().values())))
        if torch.cuda.is_available():
            self.dnn.cuda()
        self.height, self.width = tuple(map(int, config.get('image', 'size').split()))
        output = self.dnn(torch.autograd.Variable(utils.ensure_device(torch.zeros(1, 3, self.height, self.width))))
        _, _, self.rows, self.cols = output.size()
        self.i, self.j = self.rows // 2, self.cols // 2
        self.output = output[:, :, self.i, self.j]
        dataset = Dataset(self.height, self.width)
        try:
            workers = self.config.getint('data', 'workers')
        except configparser.NoOptionError:
            workers = multiprocessing.cpu_count()
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, num_workers=workers)

    def __call__(self):
        changed = np.zeros([self.height, self.width], np.bool)
        for yx in tqdm.tqdm(self.loader):
            batch_size = yx.size(0)
            tensor = torch.zeros(batch_size, 3, self.height, self.width)
            for i, _yx in enumerate(torch.unbind(yx)):
                y, x = torch.unbind(_yx)
                tensor[i, :, y, x] = 1
            tensor = utils.ensure_device(tensor)
            output = self.dnn(torch.autograd.Variable(tensor))
            output = output[:, :, self.i, self.j]
            cmp = output == self.output
            cmp = torch.prod(cmp, -1).data
            for _yx, c in zip(torch.unbind(yx), torch.unbind(cmp)):
                y, x = torch.unbind(_yx)
                changed[y, x] = c
        return changed


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    analyzer = Analyzer(args, config)
    changed = analyzer()
    os.makedirs(analyzer.model_dir, exist_ok=True)
    path = os.path.join(analyzer.model_dir, args.filename)
    scipy.misc.imsave(path, (~changed).astype(np.uint8) * 255)
    logging.info(path)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('-n', '--filename', default='receptive_field.jpg')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
