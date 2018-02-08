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

import logging
import itertools
import inspect

import numpy as np
import torch
import scipy.misc
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import humanize
import graphviz
import cv2

import utils


class DrawBBox(object):
    def __init__(self, config, category, colors=None):
        self.config = config
        if colors is None:
            self.colors = [tuple(map(lambda c: c * 255, matplotlib.colors.colorConverter.to_rgb(prop['color'])[::-1])) for prop in plt.rcParams['axes.prop_cycle']]
        else:
            self.colors = [tuple(map(lambda c: c * 255, matplotlib.colors.colorConverter.to_rgb(c)[::-1])) for c in colors]
        self.category = category

    def __call__(self, image, yx_min, yx_max, cls=None, colors=None, thickness=1, line_type=cv2.LINE_8, shift=0, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, debug=False):
        colors = self.colors if colors is None else [tuple(map(lambda c: c * 255, matplotlib.colors.colorConverter.to_rgb(c)[::-1])) for c in colors]
        if cls is None:
            cls = [None] * len(yx_min)
        for color, (ymin, xmin), (ymax, xmax), cls in zip(itertools.cycle(colors), yx_min, yx_max, cls):
            try:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=thickness, lineType=line_type, shift=shift)
                if cls is not None:
                    cv2.putText(image, self.category[cls], (xmin, ymin), font_face, font_scale, color=color)
            except OverflowError as e:
                logging.warning(e, (xmin, ymin), (xmax, ymax))
        if debug:
            cv2.imshow('', image)
            cv2.waitKey(0)
        return image


class DrawIou(object):
    def __init__(self, config, alpha=0.5, cmap=None):
        self.config = config
        self.alpha = alpha
        self.cm = matplotlib.cm.get_cmap(cmap)

    def __call__(self, image, iou, debug=False):
        _iou = (iou * self.cm.N).astype(np.int)
        heatmap = self.cm(_iou)[:, :, :3] * 255
        heatmap = scipy.misc.imresize(heatmap, image.shape[:2], interp='nearest')
        canvas = (image * (1 - self.alpha) + heatmap * self.alpha).astype(np.uint8)
        if debug:
            cv2.imshow('iou_max=%f, iou_sum=%f' % (np.max(iou), np.sum(iou)), canvas)
            cv2.waitKey(0)
        return canvas


class Graph(object):
    def __init__(self, config, state_dict, cmap=None):
        self.dot = graphviz.Digraph(node_attr=dict(config.items('digraph_node_attr')), graph_attr=dict(config.items('digraph_graph_attr')))
        self.dot.format = config.get('graph', 'format')
        self.state_dict = state_dict
        self.var_name = {t._cdata: k for k, t in state_dict.items()}
        self.seen = set()
        self.index = 0
        self.drawn = set()
        self.cm = matplotlib.cm.get_cmap(cmap)
        self.metric = eval(config.get('graph', 'metric'))
        metrics = [self.metric(t) for t in state_dict.values()]
        self.minmax = [min(metrics), max(metrics)]

    def __call__(self, node):
        if node not in self.seen:
            self.traverse_next(node)
            self.traverse_tensor(node)
            self.seen.add(node)
            self.index += 1

    def traverse_next(self, node):
        if hasattr(node, 'next_functions'):
            for n, _ in node.next_functions:
                if n is not None:
                    self.__call__(n)
                    self._draw_node_edge(node, n)
        self._draw_node(node)

    def traverse_tensor(self, node):
        tensors = [t for name, t in inspect.getmembers(node) if torch.is_tensor(t)]
        if hasattr(node, 'saved_tensors'):
            tensors += node.saved_tensors
        for tensor in tensors:
            name = self.var_name[tensor._cdata]
            self.drawn.add(name)
            self._draw_tensor(node, tensor)

    def _draw_node(self, node):
        if hasattr(node, 'variable'):
            tensor = node.variable.data
            name = self.var_name[tensor._cdata]
            label = '\n'.join(map(str, [
                '%d: %s' % (self.index, name),
                list(tensor.size()),
                humanize.naturalsize(tensor.numpy().nbytes),
            ]))
            fillcolor, fontcolor = self._tensor_color(tensor)
            self.dot.node(str(id(node)), label, shape='note', fillcolor=fillcolor, fontcolor=fontcolor)
            self.drawn.add(name)
        else:
            self.dot.node(str(id(node)), '%d: %s' % (self.index, type(node).__name__), fillcolor='white')

    def _draw_node_edge(self, node, n):
        if hasattr(n, 'variable'):
            self.dot.edge(str(id(n)), str(id(node)), arrowhead='none', arrowtail='none')
        else:
            self.dot.edge(str(id(n)), str(id(node)))

    def _draw_tensor(self, node, tensor):
        name = self.var_name[tensor._cdata]
        label = '\n'.join(map(str, [
            name,
            list(tensor.size()),
            humanize.naturalsize(tensor.numpy().nbytes),
        ]))
        fillcolor, fontcolor = self._tensor_color(tensor)
        self.dot.node(name, label, style='filled, rounded', fillcolor=fillcolor, fontcolor=fontcolor)
        self.dot.edge(name, str(id(node)), style='dashed', arrowhead='none', arrowtail='none')

    def _tensor_color(self, tensor):
        level = self._norm(self.metric(tensor))
        fillcolor = self.cm(np.int(level * self.cm.N))
        fontcolor = self.cm(self.cm.N if level < 0.5 else 0)
        return matplotlib.colors.to_hex(fillcolor), matplotlib.colors.to_hex(fontcolor)

    def _norm(self, metric):
        min, max = self.minmax
        assert min <= metric <= max, (metric, self.minmax)
        if min < max:
            return (metric - min) / (max - min)
        else:
            return metric
