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

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

import utils.iou.torch


def output_channels(num_anchors, num_cls):
    if num_cls > 1:
        return num_anchors * (5 + num_cls)
    else:
        return num_anchors * 5


def meshgrid(rows, cols, swap=False):
    i = torch.arange(0, rows).repeat(cols).view(-1, 1)
    j = torch.arange(0, cols).view(-1, 1).repeat(1, rows).view(-1, 1)
    return torch.cat([i, j], 1) if swap else torch.cat([j, i], 1)


def iou_match(yx_min, yx_max, data):
    batch_size, cells, num_anchors, _ = yx_min.size()
    iou_matrix = utils.iou.torch.batch_iou_matrix(yx_min.view(batch_size, -1, 2), yx_max.view(batch_size, -1, 2), data['yx_min'], data['yx_max'])
    iou_matrix = iou_matrix.view(batch_size, cells, num_anchors, -1)
    iou, index = iou_matrix.max(-1)
    _index = torch.unbind(index.view(batch_size, -1))
    _data = {}
    for key in 'yx_min, yx_max, cls'.split(', '):
        t = data[key]
        if len(t.size()) == 2:
            t = torch.stack([d[i] for d, i in zip(torch.unbind(t, 0), _index)]).view(batch_size, cells, num_anchors)
        elif len(t.size()) == 3:
            t = torch.stack([d[i] for d, i in zip(torch.unbind(t, 0), _index)]).view(batch_size, cells, num_anchors, -1)
        _data[key] = t
    return iou_matrix, iou, index, _data


def fit_positive(rows, cols, yx_min, yx_max, anchors):
    device_id = anchors.get_device() if torch.cuda.is_available() else None
    batch_size, num, _ = yx_min.size()
    num_anchors, _ = anchors.size()
    valid = torch.prod(yx_min < yx_max, -1)
    center = (yx_min + yx_max) / 2
    ij = torch.floor(center)
    i, j = torch.unbind(ij.long(), -1)
    index = i * cols + j
    anchors2 = anchors / 2
    iou_matrix = utils.iou.torch.iou_matrix((yx_min - center).view(-1, 2), (yx_max - center).view(-1, 2), -anchors2, anchors2).view(batch_size, -1, num_anchors)
    iou, index_anchor = iou_matrix.max(-1)
    _positive = []
    cells = rows * cols
    for valid, index, index_anchor in zip(torch.unbind(valid), torch.unbind(index), torch.unbind(index_anchor)):
        index, index_anchor = (t[valid] for t in (index, index_anchor))
        t = utils.ensure_device(torch.ByteTensor(cells, num_anchors).zero_(), device_id)
        t[index, index_anchor] = 1
        _positive.append(t)
    return torch.stack(_positive)


def fill_norm(yx_min, yx_max, anchors):
    center = (yx_min + yx_max) / 2
    ij = torch.floor(center)
    center_offset = center - ij
    size = yx_max - yx_min
    return center_offset, torch.log(size / anchors.view(1, -1, 2))


def square(t):
    return t * t


class Inference(nn.Module):
    def __init__(self, config, dnn, anchors):
        nn.Module.__init__(self)
        self.config = config
        self.dnn = dnn
        self.anchors = anchors

    def forward(self, x):
        device_id = x.get_device() if torch.cuda.is_available() else None
        feature = self.dnn(x)
        rows, cols = feature.size()[-2:]
        cells = rows * cols
        _feature = feature.permute(0, 2, 3, 1).contiguous().view(feature.size(0), cells, self.anchors.size(0), -1)
        sigmoid = F.sigmoid(_feature[:, :, :, :3])
        iou = sigmoid[:, :, :, 0]
        ij = torch.autograd.Variable(utils.ensure_device(meshgrid(rows, cols).view(1, -1, 1, 2), device_id))
        center_offset = sigmoid[:, :, :, 1:3]
        center = ij + center_offset
        size_norm = _feature[:, :, :, 3:5]
        anchors = torch.autograd.Variable(utils.ensure_device(self.anchors.view(1, 1, -1, 2), device_id))
        size = torch.exp(size_norm) * anchors
        size2 = size / 2
        yx_min = center - size2
        yx_max = center + size2
        logits = _feature[:, :, :, 5:] if _feature.size(-1) > 5 else None
        return feature, iou, center_offset, size_norm, yx_min, yx_max, logits


def loss(anchors, data, pred, threshold):
    iou = pred['iou']
    device_id = iou.get_device() if torch.cuda.is_available() else None
    rows, cols = pred['feature'].size()[-2:]
    iou_matrix, _iou, _, _data = iou_match(pred['yx_min'].data, pred['yx_max'].data, data)
    anchors = utils.ensure_device(anchors, device_id)
    positive = fit_positive(rows, cols, *(data[key] for key in 'yx_min, yx_max'.split(', ')), anchors)
    negative = ~positive & (_iou < threshold)
    _center_offset, _size_norm = fill_norm(*(_data[key] for key in 'yx_min, yx_max'.split(', ')), anchors)
    positive, negative, _iou, _center_offset, _size_norm, _cls = (torch.autograd.Variable(t) for t in (positive, negative, _iou, _center_offset, _size_norm, _data['cls']))
    _positive = torch.unsqueeze(positive, -1)
    loss = {}
    # iou
    loss['foreground'] = F.mse_loss(iou[positive], _iou[positive], size_average=False)
    loss['background'] = torch.sum(square(iou[negative]))
    # bbox
    loss['center'] = F.mse_loss(pred['center_offset'][_positive], _center_offset[_positive], size_average=False)
    loss['size'] = F.mse_loss(pred['size_norm'][_positive], _size_norm[_positive], size_average=False)
    # cls
    if 'logits' in pred:
        logits = pred['logits']
        if len(_cls.size()) > 3:
            loss['cls'] = F.mse_loss(F.softmax(logits, -1)[_positive], _cls[_positive], size_average=False)
        else:
            loss['cls'] = F.cross_entropy(logits[_positive].view(-1, logits.size(-1)), _cls[positive].view(-1))
    # normalize
    cnt = float(np.multiply.reduce(positive.size()))
    for key in loss:
        loss[key] /= cnt
    return loss, dict(iou=_iou, data=_data, positive=positive, negative=negative)


def _inference(inference, tensor):
    feature, iou, center_offset, size_norm, yx_min, yx_max, logits = inference(tensor)
    pred = dict(
        feature=feature, iou=iou,
        center_offset=center_offset, size_norm=size_norm,
        yx_min=yx_min, yx_max=yx_max,
    )
    if logits is not None:
        pred['logits'] = logits.contiguous()
    return pred
