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

import collections

import torch.nn as nn

import model


def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(collections.OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('act', nn.ReLU(inplace=True)),
    ]))


def conv_dw(in_channels, stride):
    return nn.Sequential(collections.OrderedDict([
        ('conv', nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)),
        ('bn', nn.BatchNorm2d(in_channels)),
        ('act', nn.ReLU(inplace=True)),
    ]))


def conv_pw(in_channels, out_channels):
    return nn.Sequential(collections.OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('act', nn.ReLU(inplace=True)),
    ]))


def conv_unit(in_channels, out_channels, stride):
    return nn.Sequential(collections.OrderedDict([
        ('dw', conv_dw(in_channels, stride)),
        ('pw', conv_pw(in_channels, out_channels)),
    ]))


class MobileNet(nn.Module):
    def __init__(self, config, anchors, num_cls):
        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_unit(32, 64, 1),
            conv_unit(64, 128, 2),
            conv_unit(128, 128, 1),
            conv_unit(128, 256, 2),
            conv_unit(256, 256, 1),
            conv_unit(256, 512, 2),
            conv_unit(512, 512, 1),
            conv_unit(512, 512, 1),
            conv_unit(512, 512, 1),
            conv_unit(512, 512, 1),
            conv_unit(512, 512, 1),
            conv_unit(512, 1024, 2),
            conv_unit(1024, 1024, 1),
            nn.Conv2d(1024, model.output_channels(len(anchors), num_cls), 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)
