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

import numpy as np
import scipy.misc
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import cv2


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
