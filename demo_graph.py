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
import yaml

import numpy as np
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import matplotlib.cm
import matplotlib.colors
import humanize
import graphviz

import utils.train


class Graph(object):
    def __init__(self, config, state_dict, cmap=None):
        self.dot = graphviz.Digraph(node_attr=dict(config.items('digraph_node_attr')), graph_attr=dict(config.items('digraph_graph_attr')))
        self.dot.format = config.get('graph', 'format')
        self.state_dict = state_dict
        self.var_name = {t._cdata: k for k, t in state_dict.items()}
        self.seen = set()
        self.drawn = set()
        self.cm = matplotlib.cm.get_cmap(cmap)
        self.metric = eval(config.get('graph', 'metric'))
        metrics = [self.metric(t.numpy()) for t in state_dict.values()]
        self.minmax = [min(metrics), max(metrics)]

    def __call__(self, node):
        if node not in self.seen:
            self.seen.add(node)
            if hasattr(node, 'next_functions'):
                for child, _ in node.next_functions:
                    if child is not None:
                        if self.__call__(child) is None:
                            self.dot.edge(str(id(child)), str(id(node)))
                        else:
                            self.dot.edge(str(id(child)), str(id(node)), arrowhead='none', arrowtail='none')
            tensors = [t for name, t in inspect.getmembers(node) if torch.is_tensor(t)]
            if hasattr(node, 'saved_tensors'):
                tensors += node.saved_tensors
            for tensor in tensors:
                name = self.draw_tensor(tensor, style='filled, rounded')
                self.dot.edge(name, str(id(node)), style='dashed', arrowhead='none', arrowtail='none')
                self.drawn.add(name)
            return self.draw_node(node)

    def draw_node(self, node):
        if hasattr(node, 'variable'):
            return self.draw_tensor(node.variable.data, name=str(id(node)), shape='note')
        else:
            self.dot.node(str(id(node)), type(node).__name__, fillcolor='white')

    def draw_tensor(self, tensor, name=None, **kwargs):
        var_name = self.var_name[tensor._cdata]
        label = '\n'.join(map(str, [
            var_name,
            list(tensor.size()),
            humanize.naturalsize(tensor.numpy().nbytes),
        ]))
        level = self.norm(self.metric(tensor.numpy()))
        fillcolor = self.cm(np.int(level * self.cm.N))
        fontcolor = self.cm(self.cm.N if level < 0.5 else 0)
        self.dot.node(var_name if name is None else name, label, fillcolor=matplotlib.colors.to_hex(fillcolor), fontcolor=matplotlib.colors.to_hex(fontcolor), **kwargs)
        self.drawn.add(var_name)
        return var_name

    def norm(self, metric):
        min, max = self.minmax
        assert min <= metric <= max, (metric, self.minmax)
        if min < max:
            return (metric - min) / (max - min)
        else:
            return metric


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
    try:
        path, step, epoch = utils.train.load_model(model_dir)
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        dnn.load_state_dict(state_dict)
    except ValueError:
        logging.warning('model cannot be loaded')
    state_dict = dnn.state_dict()
    graph = Graph(config, state_dict)
    graph(output.grad_fn)
    diff = [key for key in state_dict if key not in graph.drawn]
    if diff:
        logging.warning('variables not shown: ' + str(diff))
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