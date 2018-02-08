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

import inspect
import collections
import enum
import copy
import unittest
import configparser
import logging

import numpy as np
import torch
import humanize
import graphviz

import model.yolo2
import utils


def prune(modifier, channels):
    for name in modifier.output:
        offset = modifier.output[name]
        var = modifier.state_dict[name]
        var = var[channels]
        logging.info('prune output channels of %s from %s to %s with offset %d' % (name, 'x'.join(map(str, modifier.state_dict[name].size())), 'x'.join(map(str, var.size())), offset))
        modifier.state_dict[name] = var
    for name in modifier.input:
        offset = modifier.input[name]
        var = modifier.state_dict[name]
        var = torch.stack([v[channels] for v in torch.unbind(var)])
        logging.info('prune input channels of %s from %s to %s with offset %d' % (name, 'x'.join(map(str, modifier.state_dict[name].size())), 'x'.join(map(str, var.size())), offset))
        modifier.state_dict[name] = var


class Mode(enum.IntEnum):
    NONE = 0
    OUTPUT = 1
    INPUT = 2


class Closure(object):
    node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.2',
    )
    graph_attr = dict(
        size='12, 12'
    )
    format = 'svg'

    def __init__(self, name, state_dict, scope=2, debug=False):
        self.name = name
        self.state_dict = state_dict
        self.scope = scope
        if debug:
            self.dot = graphviz.Digraph(node_attr=self.node_attr, graph_attr=self.graph_attr)
            self.dot.format = self.format
        self.var_name = {t._cdata: k for k, t in state_dict.items()}
        self.seen = collections.OrderedDict()
        self.index = 0
        self.output = collections.OrderedDict()
        self.input = collections.OrderedDict()

    def __call__(self, node):
        edge = dict(
            mode=Mode.NONE,
            prefix=self.prefix(self.name),
            offset=0,
        )
        return self.traverse(node, **edge)

    def traverse(self, node, **edge):
        if node in self.seen:
            return self.seen[node]
        else:
            edge = self.traverse_next(node, edge)
            if hasattr(node, 'variable'):
                name = self.var_name[node.variable.data._cdata]
                edge = self.check(name, edge)
            edge = self.traverse_tensor(node, edge)
            edge['index'] = self.index
            self.seen[node] = copy.copy(edge)
            self.index += 1
        return edge

    def traverse_next(self, node, edge):
        if hasattr(node, 'next_functions'):
            _edges = []
            _edge = edge
            for n, _ in node.next_functions:
                if n is not None:
                    _edge = self._draw_node_edge(node, n, _edge, self.traverse(n, **_edge))
                    _edges.append(_edge)
            if _edges:
                if type(node).__name__ == 'CatBackward':
                    edge['channels'] = sum(map(lambda edge: edge['channels'], _edges))
                else:
                    edge['channels'] = _edges[0]['channels']
                modes = list(map(lambda edge: edge['mode'], _edges))
                mode = max(modes)
                if mode != edge['mode']:
                    edge['mode'] = mode
                    if type(node).__name__ == 'CatBackward':
                        for e in _edges[:modes.index(mode)]:
                            edge['offset'] += e['channels']
        self._draw_node(node, edge)
        return edge

    def traverse_tensor(self, node, edge):
        tensors = [t for name, t in inspect.getmembers(node) if torch.is_tensor(t)]
        if hasattr(node, 'saved_tensors'):
            tensors += node.saved_tensors
        for tensor in tensors:
            name = self.var_name[tensor._cdata]
            edge = self.check(name, edge)
            self._draw_tensor(node, tensor, edge)
        return edge

    def prefix(self, name):
        return '.'.join(name.split('.')[:-self.scope])

    def check(self, name, edge):
        prefix = self.prefix(name)
        edge = type(self).switch_mode(prefix, edge)
        var = self.state_dict[name]
        if prefix == edge['prefix']:
            if edge['mode'] == Mode.OUTPUT:
                self.output[name] = edge['offset']
            elif edge['mode'] == Mode.INPUT:
                if len(var.size()) > 1:
                    self.input[name] = edge['offset']
        edge['channels'] = var.size(0)
        return edge

    @staticmethod
    def switch_mode(prefix, edge):
        if edge['mode'] == Mode.NONE and prefix == edge['prefix']:
            edge['mode'] = Mode.OUTPUT
        elif edge['mode'] == Mode.OUTPUT and prefix != edge['prefix']:
            edge['mode'] = Mode.INPUT
            edge['prefix'] = prefix
        return edge

    def _draw_node(self, node, edge):
        if not hasattr(self, 'dot'):
            return
        if hasattr(node, 'variable'):
            tensor = node.variable.data
            name = self.var_name[tensor._cdata]
            label = '\n'.join(map(str, [
                '%d: %s (%s)' % (self.index, name, str(edge['mode']).split('.')[-1]),
                list(tensor.size()),
                humanize.naturalsize(tensor.numpy().nbytes),
            ]))
            self.dot.node(str(id(node)), label, shape='note')
        else:
            self.dot.node(str(id(node)), '%d: %s (%s)' % (self.index, type(node).__name__, str(edge['mode']).split('.')[-1]), fillcolor='white')

    def _draw_node_edge(self, node, n, edge_before, edge_after):
        if not hasattr(self, 'dot'):
            return edge_after
        label = '%s->%s' % (str(edge_before['mode']).split('.')[-1], str(edge_after['mode']).split('.')[-1])
        if hasattr(n, 'variable'):
            self.dot.edge(str(id(n)), str(id(node)), label, arrowhead='none', arrowtail='none')
        else:
            self.dot.edge(str(id(n)), str(id(node)), label)
        return edge_after

    def _draw_tensor(self, node, tensor, edge):
        if not hasattr(self, 'dot'):
            return
        name = self.var_name[tensor._cdata]
        label = '\n'.join(map(str, [
            name,
            list(tensor.size()),
            humanize.naturalsize(tensor.numpy().nbytes),
        ]))
        self.dot.node(name, label, style='filled, rounded')
        self.dot.edge(name, str(id(node)), str(edge['mode']).split('.')[-1], style='dashed', arrowhead='none', arrowtail='none')


class TestYolo2Tiny(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        self.config_channels = model.ConfigChannels(config)
        self.category = ['test%d' % i for i in range(1)]
        self.anchors = torch.from_numpy(np.array([
            (1, 1),
            (1, 2),
        ], dtype=np.float32))
        module = model.yolo2
        self.model = module.Tiny
        size = module.settings['size']
        self.image = torch.autograd.Variable(torch.randn(1, 3, *size))

    def test_layers_14_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        closure = Closure(name, state_dict)
        closure(output.grad_fn)
        self.assertDictEqual(closure.output, {
            'layers.14.conv.weight': 0,
            'layers.14.bn.weight': 0,
            'layers.14.bn.bias': 0,
            'layers.14.bn.running_mean': 0,
            'layers.14.bn.running_var': 0,
        })
        self.assertDictEqual(closure.input, {
            'layers.15.conv.weight': 0,
        })
        d = utils.dense(state_dict[name])
        channels = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        prune(closure, channels)
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn(self.image)
        self.assertEqual(len(channels), len(dnn.state_dict()[name]))


class TestYolo2Darknet(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        self.config_channels = model.ConfigChannels(config)
        self.category = ['test%d' % i for i in range(1)]
        self.anchors = torch.from_numpy(np.array([
            (1, 1),
            (1, 2),
        ], dtype=np.float32))
        module = model.yolo2
        self.model = module.Darknet
        size = module.settings['size']
        self.image = torch.autograd.Variable(torch.randn(1, 3, *size))

    def test_layers1_0_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        closure = Closure(name, state_dict)
        closure(output.grad_fn)
        self.assertDictEqual(closure.output, {
            'layers1.0.conv.weight': 0,
            'layers1.0.bn.weight': 0,
            'layers1.0.bn.bias': 0,
            'layers1.0.bn.running_mean': 0,
            'layers1.0.bn.running_var': 0,
        })
        self.assertDictEqual(closure.input, {
            'layers1.2.conv.weight': 0,
        })
        d = utils.dense(state_dict[name])
        channels = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        prune(closure, channels)
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn(self.image)
        self.assertEqual(len(channels), len(dnn.state_dict()[name]))

    def test_layers1_16_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        closure = Closure(name, state_dict)
        closure(output.grad_fn)
        self.assertDictEqual(closure.output, {
            'layers1.16.conv.weight': 0,
            'layers1.16.bn.weight': 0,
            'layers1.16.bn.bias': 0,
            'layers1.16.bn.running_mean': 0,
            'layers1.16.bn.running_var': 0,
        })
        self.assertDictEqual(closure.input, {
            'passthrough.conv.weight': 0,
            'layers2.1.conv.weight': 0,
        })
        d = utils.dense(state_dict[name])
        channels = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        prune(closure, channels)
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn(self.image)
        self.assertEqual(len(channels), len(dnn.state_dict()[name]))

    def test_passthrough_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        closure = Closure(name, state_dict)
        closure(output.grad_fn)
        self.assertDictEqual(closure.output, {
            'passthrough.conv.weight': 0,
            'passthrough.bn.weight': 0,
            'passthrough.bn.bias': 0,
            'passthrough.bn.running_mean': 0,
            'passthrough.bn.running_var': 0,
        })
        self.assertDictEqual(closure.input, {
            'layers3.0.conv.weight': 0,
        })
        d = utils.dense(state_dict[name])
        channels = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        prune(closure, channels)
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn(self.image)
        self.assertEqual(len(channels), len(dnn.state_dict()[name]))

    def test_layers2_1_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        closure = Closure(name, state_dict)
        closure(output.grad_fn)
        self.assertDictEqual(closure.output, {
            'layers2.1.conv.weight': 0,
            'layers2.1.bn.weight': 0,
            'layers2.1.bn.bias': 0,
            'layers2.1.bn.running_mean': 0,
            'layers2.1.bn.running_var': 0,
        })
        self.assertDictEqual(closure.input, {
            'layers2.2.conv.weight': 0,
        })
        d = utils.dense(state_dict[name])
        channels = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        prune(closure, channels)
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn(self.image)
        self.assertEqual(len(channels), len(dnn.state_dict()[name]))

    def test_layers2_7_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        closure = Closure(name, state_dict)
        closure(output.grad_fn)
        self.assertDictEqual(closure.output, {
            'layers2.7.conv.weight': 0,
            'layers2.7.bn.weight': 0,
            'layers2.7.bn.bias': 0,
            'layers2.7.bn.running_mean': 0,
            'layers2.7.bn.running_var': 0,
        })
        self.assertDictEqual(closure.input, {
            'layers3.0.conv.weight': 64,
        })
        d = utils.dense(state_dict[name])
        channels = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        prune(closure, channels)
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn(self.image)
        self.assertEqual(len(channels), len(dnn.state_dict()[name]))


if __name__ == '__main__':
    unittest.main()
