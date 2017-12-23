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

import torch

import utils.iou.torch


def nms(yx_min, yx_max, score, overlap=0.5, limit=200):
    """
    Filtering the overlapping (IoU > overlap threshold) bounding boxes according to the score (in descending order).
    :author 申瑞珉 (Ruimin Shen)
    :param yx_min: The top left coordinates (y, x) of the list (size [N, 2]) of bounding boxes.
    :param yx_max: The bottom right coordinates (y, x) of the list (size [N, 2]) of bounding boxes.
    :param score: The scores of the list (size [N]) of bounding boxes.
    :param overlap: The IoU threshold.
    :param limit: Limits the number of results.
    :return: The indices of the selected bounding boxes.
    """
    keep = []
    if score.numel() == 0:
        return keep
    s, index = score.sort(0)
    index = index[-limit:]
    while index.numel() > 0:
        i = index[-1]
        keep.append(i)
        if index.size(0) == 1:
            break
        index = index[:-1]
        yx_min1, yx_max1 = (torch.unsqueeze(t[i], 0) for t in (yx_min, yx_max))
        yx_min2, yx_max2 = (torch.index_select(t, 0, index) for t in (yx_min, yx_max))
        iou = utils.iou.torch.iou_matrix(yx_min1, yx_max1, yx_min2, yx_max2)[0]
        index = index[iou <= overlap]
    return keep
