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
import humanize
import graphviz

import utils.train
import model


class Graph(object):
    def __init__(self, config, state_dict):
        self.seen = set()
        self.mapper = {id(v): k for k, v in state_dict.items()}
        self.dot = graphviz.Digraph(node_attr=dict(config.items('digraph_node_attr')), graph_attr=dict(config.items('digraph_graph_attr')))

    def __call__(self, fn):
        if fn not in self.seen:
            if torch.is_tensor(fn):
                self.dot.node(str(id(fn)), ', '.join(map(str, fn.size())), fillcolor='orange')
            elif hasattr(fn, 'variable'):
                var = fn.variable
                node_name = '%s\n %s' % (self.mapper[id(var.data)], ', '.join(map(str, var.size())))
                self.dot.node(str(id(fn)), node_name, fillcolor='lightblue')
            else:
                self.dot.node(str(id(fn)), type(fn).__name__)
            self.seen.add(fn)
            if hasattr(fn, 'next_functions'):
                for _fn, _ in fn.next_functions:
                    if _fn is not None:
                        self.dot.edge(str(id(_fn)), str(id(fn)))
                        self.__call__(_fn)
            if hasattr(fn, 'saved_tensors'):
                for t in fn.saved_tensors:
                    self.dot.edge(str(id(t)), str(id(fn)))
                    self.__call__(t)


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
    dnn = utils.parse_attr(config.get('model', 'dnn'))(config, anchors, len(category))
    logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in dnn.state_dict().values())))
    height, width = tuple(map(int, config.get('image', 'size').split()))
    image = torch.autograd.Variable(torch.randn(args.batch_size, 3, height, width))
    output = dnn(image)
    graph = Graph(config, dnn.state_dict())
    graph(output.grad_fn)
    path = graph.dot.view(directory=model_dir)
    logging.info(path)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
