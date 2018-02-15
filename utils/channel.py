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
import copy
import unittest
import configparser

import numpy as np
import torch
import humanize
import graphviz

import model.inception4
import model.yolo2
import utils


class Modifier(object):
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

    def __init__(self, name, state_dict, model, modify_output, modify_input, debug=False):
        self.name = name
        self.state_dict = state_dict
        self.model = model
        self.modify_output = modify_output
        self.modify_input = modify_input
        if debug:
            self.dot = graphviz.Digraph(node_attr=self.node_attr, graph_attr=self.graph_attr)
            self.dot.format = self.format
        self.var_name = {t._cdata: k for k, t in state_dict.items()}
        self.seen = collections.OrderedDict()
        self.index = 0

    def __call__(self, node):
        edge = dict(
            scope=self.model.scope(self.name),
        )
        return self.traverse(node, **edge)

    def traverse(self, node, **edge_init):
        if node in self.seen:
            return self.seen[node]
        else:
            inputs, edge = self.merge_edges(node, edge_init)
            edge = self.traverse_vars(node, edge)
            edge['index'] = self.index
            if hasattr(self, 'dot'):
                for input in inputs:
                    self._draw_node_edge(node, input)
                self._draw_node(node, edge)
            edge = copy.deepcopy(edge)
            self.seen[node] = edge
            self.index += 1
        return edge

    def merge_edges(self, node, edge_init):
        edge = copy.deepcopy(edge_init)
        edge_init.pop('_modify', None)
        if hasattr(node, 'next_functions'):
            inputs = []
            for _node, _ in node.next_functions:
                if _node is not None:
                    _edge = self.traverse(_node, **edge_init)
                    if 'modify' in _edge:
                        edge_init['_modify'] = copy.deepcopy(_edge['modify'])
                    inputs.append(dict(
                        node=_node,
                        edge=_edge,
                    ))
            if inputs:
                for key in 'channels, _channels'.split(', '):
                    edge[key] = self.merge_channels(node, inputs, key)
                self.merge_modify(node, inputs, edge)
        return inputs, edge

    def merge_channels(self, node, inputs, key):
        name = type(node).__name__
        if name == 'CatBackward':
            channels = sum(map(lambda input: input['edge'][key], inputs))
        else:
            channels = inputs[-1]['edge'][key]
        if hasattr(self.model, 'get_mapper'):
            mapper = self.model.get_mapper(self.index)
            if mapper is not None:
                indices = torch.LongTensor(np.arange(channels))
                indices = mapper(indices, channels)
                channels = len(indices)
        return channels

    def merge_modify(self, node, inputs, edge):
        for index, input in enumerate(inputs):
            _edge = input['edge']
            if 'modify' in _edge:
                modify = copy.deepcopy(_edge['modify'])
                if 'modify' in edge:
                    if 'scope' in modify:
                        edge['modify']['scope'] = modify['scope']
                else:
                    begin, end = modify['range']
                    name = type(node).__name__
                    if name == 'CatBackward':
                        offset = sum(map(lambda input: input['edge']['_channels'], inputs[:index]))
                        begin += offset
                        end += offset
                    if hasattr(self.model, 'get_mapper'):
                        mapper = self.model.get_mapper(self.index)
                        if mapper is not None:
                            channels = end - begin
                            indices = torch.LongTensor(np.arange(channels))
                            indices = mapper(indices, channels)
                            channels = len(indices)
                            end = begin + channels
                            modify['mappers'].append(mapper)
                    modify['range'] = (begin, end)
                    edge['modify'] = modify

    def traverse_vars(self, node, edge):
        if hasattr(node, 'variable'):
            name = self.var_name[node.variable.data._cdata]
            edge = self.modify(name, edge)
        tensors = [t for name, t in inspect.getmembers(node) if torch.is_tensor(t)]
        if hasattr(node, 'saved_tensors'):
            tensors += node.saved_tensors
        for tensor in tensors:
            name = self.var_name[tensor._cdata]
            edge = self.modify(name, edge)
            if hasattr(self, 'dot'):
                self._draw_tensor(node, tensor, edge)
        return edge

    def modify(self, name, edge):
        scope = self.model.scope(name)
        var = self.state_dict[name]
        edge['_channels'] = var.size(0)
        if scope == edge['scope']:
            edge['_size'] = var.size()
            edge['modify'] = dict(
                range=(0, var.size(0)),
                mappers=[],
            )
            var = self.modify_output(name, var)
        elif '_modify' in edge:
            modify = edge['_modify']
            if 'scope' not in modify:
                modify['scope'] = scope
            if 'scope' in modify and modify['scope'] == scope:
                if len(var.size()) > 1:
                    edge['_size'] = var.size()
                    begin, end = modify['range']
                    def mapper(indices, channels):
                        for m in modify['mappers']:
                            indices = m(indices, channels)
                        return indices
                    vars = []
                    for v in torch.unbind(var):
                        comp = []
                        if begin > 0:
                            comp.append(v[:begin])
                        _v = v[begin:end]
                        _v = self.modify_input(name, _v, mapper)
                        comp.append(_v)
                        if end < var.size(1):
                            comp.append(v[end:])
                        v = torch.cat(comp)
                        vars.append(v)
                    var = torch.stack(vars)
                    edge['modify'] = modify
            else:
                edge.pop('modify', None)
        edge['channels'] = var.size(0)
        self.state_dict[name] = var
        return edge

    def _draw_node(self, node, edge):
        if hasattr(node, 'variable'):
            name = self.var_name[node.variable.data._cdata]
            tensor = self.state_dict[name]
            label = '\n'.join(map(str, filter(lambda x: x is not None, [
                '%d: %s' % (self.index, name),
                type(self)._pretty_size(tensor.size(), edge),
                humanize.naturalsize(tensor.numpy().nbytes),
            ])))
            self.dot.node(str(id(node)), label, shape='note')
        else:
            name = type(node).__name__
            label = '%d: %s' % (self.index, name)
            self.dot.node(str(id(node)), label, fillcolor='white')

    def _draw_node_edge(self, node, input):
        _node = input['node']
        edge = input['edge']
        channels, _channels = (edge[key] for key in 'channels, _channels'.split(', '))
        label = '\n'.join(map(str, filter(lambda x: x is not None, [
            channels if channels == _channels else '%d->%d' % (_channels, channels),
            type(self)._pretty_modify(edge),
        ])))
        if hasattr(_node, 'variable'):
            self.dot.edge(str(id(_node)), str(id(node)), label, arrowhead='none', arrowtail='none')
        else:
            self.dot.edge(str(id(_node)), str(id(node)), label)

    def _draw_tensor(self, node, tensor, edge):
        name = self.var_name[tensor._cdata]
        tensor = self.state_dict[name]
        label = '\n'.join(map(str, filter(lambda x: x is not None, [
            name,
            type(self)._pretty_size(tensor.size(), edge),
            humanize.naturalsize(tensor.numpy().nbytes),
        ])))
        self.dot.node(name, label, style='filled, rounded')
        channels, _channels = (edge[key] for key in 'channels, _channels'.split(', '))
        label = '\n'.join(map(str, filter(lambda x: x is not None, [
            channels if channels == _channels else '%d->%d' % (_channels, channels),
            type(self)._pretty_modify(edge),
        ])))
        self.dot.edge(name, str(id(node)), label, style='dashed', arrowhead='none', arrowtail='none')

    @staticmethod
    def _pretty_size(size, edge):
        if '_size' in edge:
            comp = []
            for _s, s in zip(edge['_size'], size):
                if s == _s:
                    content = str(s)
                elif 'range' in edge:
                    begin, end = edge['range']
                    content = '%d[%d:%d]->%d' % (_s, begin, end, s)
                else:
                    content = '%d->%d' % (_s, s)
                comp.append(content)
            return ', '.join(comp)
        else:
            return ', '.join(map(str, size))

    @staticmethod
    def _pretty_modify(edge):
        if 'modify' in edge:
            modify = edge['modify']
            mode = 'input' if 'scope' in modify else 'output'
            begin, end = modify['range']
            return '%s[%d:%d]' % (mode, begin, end)


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
        d = utils.dense(state_dict[name])
        keep = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        modifier = Modifier(
            name, state_dict, dnn,
            lambda name, var: var[keep],
            lambda name, var, mapper: var[mapper(keep, len(d))],
        )
        modifier(output.grad_fn)
        # check channels
        scope = dnn.scope(name)
        self.assertEqual(state_dict[name].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.weight'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.bias'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_mean'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_var'].size(0), len(keep))
        # check if runnable
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        dnn(self.image)


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
        d = utils.dense(state_dict[name])
        keep = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        modifier = Modifier(
            name, state_dict, dnn,
            lambda name, var: var[keep],
            lambda name, var, mapper: var[mapper(keep, len(d))],
        )
        modifier(output.grad_fn)
        # check channels
        scope = dnn.scope(name)
        self.assertEqual(state_dict[name].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.weight'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.bias'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_mean'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_var'].size(0), len(keep))
        # check if runnable
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        dnn(self.image)

    def test_layers1_16_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        d = utils.dense(state_dict[name])
        keep = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        modifier = Modifier(
            name, state_dict, dnn,
            lambda name, var: var[keep],
            lambda name, var, mapper: var[mapper(keep, len(d))],
        )
        modifier(output.grad_fn)
        # check channels
        scope = dnn.scope(name)
        self.assertEqual(state_dict[name].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.weight'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.bias'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_mean'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_var'].size(0), len(keep))
        # check if runnable
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        dnn(self.image)

    def test_passthrough_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        d = utils.dense(state_dict[name])
        keep = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        modifier = Modifier(
            name, state_dict, dnn,
            lambda name, var: var[keep],
            lambda name, var, mapper: var[mapper(keep, len(d))],
        )
        modifier(output.grad_fn)
        # check channels
        scope = dnn.scope(name)
        self.assertEqual(state_dict[name].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.weight'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.bias'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_mean'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_var'].size(0), len(keep))
        # check if runnable
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        dnn(self.image)

    def test_layers2_1_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        d = utils.dense(state_dict[name])
        keep = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        modifier = Modifier(
            name, state_dict, dnn,
            lambda name, var: var[keep],
            lambda name, var, mapper: var[mapper(keep, len(d))],
        )
        modifier(output.grad_fn)
        # check channels
        scope = dnn.scope(name)
        self.assertEqual(state_dict[name].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.weight'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.bias'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_mean'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_var'].size(0), len(keep))
        # check if runnable
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        dnn(self.image)

    def test_layers2_7_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        d = utils.dense(state_dict[name])
        keep = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        modifier = Modifier(
            name, state_dict, dnn,
            lambda name, var: var[keep],
            lambda name, var, mapper: var[mapper(keep, len(d))],
        )
        modifier(output.grad_fn)
        # check channels
        scope = dnn.scope(name)
        self.assertEqual(state_dict[name].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.weight'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.bias'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_mean'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_var'].size(0), len(keep))
        # check if runnable
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        dnn(self.image)

    def test_layers3_0_conv_weight(self):
        dnn = self.model(self.config_channels, self.anchors, len(self.category))
        output = dnn(self.image)
        state_dict = dnn.state_dict()
        name = '.'.join(self.id().split('.')[-1].split('_')[1:])
        d = utils.dense(state_dict[name])
        keep = torch.LongTensor(np.argsort(d)[int(len(d) * 0.5):])
        modifier = Modifier(
            name, state_dict, dnn,
            lambda name, var: var[keep],
            lambda name, var, mapper: var[mapper(keep, len(d))],
        )
        modifier(output.grad_fn)
        # check channels
        scope = dnn.scope(name)
        self.assertEqual(state_dict[name].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.weight'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.bias'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_mean'].size(0), len(keep))
        self.assertEqual(state_dict[scope + '.bn.running_var'].size(0), len(keep))
        # check if runnable
        config_channels = model.ConfigChannels(self.config_channels.config, state_dict)
        dnn = self.model(config_channels, self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        dnn(self.image)


if __name__ == '__main__':
    unittest.main()
