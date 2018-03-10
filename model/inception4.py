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
import configparser
import collections.abc

import torch
import torch.nn as nn
from pretrainedmodels.models.inceptionv4 import pretrained_settings

import model


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bn=True, act=True):
        nn.Module.__init__(self)
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True) if bn else lambda x: x
        self.act = nn.ReLU(inplace=True) if act else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Mixed_3a(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.conv.conv.weight' % prefix), kernel_size=3, stride=2, bn=bn)
        config_channels.channels = channels + self.conv.conv.weight.size(0)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        # branch0
        channels = config_channels.channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(64 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=1, bn=bn))
        self.branch0 = nn.Sequential(*branch)
        # branch1
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=(3, 3), stride=1, bn=bn))
        self.branch1 = nn.Sequential(*branch)
        # output
        config_channels.channels = self.branch0[-1].conv.weight.size(0) + self.branch1[-1].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.conv = Conv2d(config_channels.channels, config_channels(int(192 * ratio), '%s.conv.conv.weight' % prefix), kernel_size=3, stride=2, bn=bn)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        config_channels.channels = self.conv.conv.weight.size(0) + channels

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=1, stride=1, bn=bn)
        # branch1
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=1, padding=1, bn=bn))
        self.branch1 = nn.Sequential(*branch)
        # branch2
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(64 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=1, padding=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=1, padding=1, bn=bn))
        self.branch2 = nn.Sequential(*branch)
        #branch3
        config_channels.channels = channels
        branch = []
        branch.append(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False))
        branch.append(Conv2d(config_channels.channels, config_channels(int(96 * ratio), '%s.branch3.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1))
        self.branch3 = nn.Sequential(*branch)
        # output
        config_channels.channels = self.branch0.conv.weight.size(0) + self.branch1[-1].conv.weight.size(0) + self.branch2[-1].conv.weight.size(0) + self.branch3[-1].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int(384 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=3, stride=2, bn=bn)
        # branch1
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(192 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(224 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=1, padding=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=2, bn=bn))
        self.branch1 = nn.Sequential(*branch)

        self.branch2 = nn.MaxPool2d(3, stride=2)
        # output
        config_channels.channels = self.branch0.conv.weight.size(0) + self.branch1[-1].conv.weight.size(0) + channels

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int(384 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=1, stride=1, bn=bn)
        # branch1
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(192 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(224 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        self.branch1 = nn.Sequential(*branch)
        # branch2
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(192 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(192 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(224 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(224 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(256 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        self.branch2 = nn.Sequential(*branch)
        # branch3
        config_channels.channels = channels
        branch = []
        branch.append(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False))
        branch.append(Conv2d(config_channels.channels, config_channels(int(128 * ratio), '%s.branch3.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        self.branch3 = nn.Sequential(*branch)
        # output
        config_channels.channels = self.branch0.conv.weight.size(0) + self.branch1[-1].conv.weight.size(0) + self.branch2[-1].conv.weight.size(0) + self.branch3[-1].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        # branch0
        channels = config_channels.channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(192 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(192 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=2, bn=bn))
        self.branch0 = nn.Sequential(*branch)
        # branch1
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(320 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(320 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch))), kernel_size=3, stride=2, bn=bn))
        self.branch1 = nn.Sequential(*branch)
        self.branch2 = nn.MaxPool2d(3, stride=2)
        # output
        config_channels.channels = self.branch0[-1].conv.weight.size(0) + self.branch1[-1].conv.weight.size(0) + channels

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):
    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int(256 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=1, stride=1, bn=bn)
        # branch1
        config_channels.channels = channels
        self.branch1_0 = Conv2d(config_channels.channels, config_channels(int(384 * ratio), '%s.branch1_0.conv.weight' % prefix), kernel_size=1, stride=1, bn=bn)
        _channels = config_channels.channels
        self.branch1_1a = Conv2d(_channels, config_channels(int(256 * ratio), '%s.branch1_1a.conv.weight' % prefix), kernel_size=(1, 3), stride=1, padding=(0, 1), bn=bn)
        self.branch1_1b = Conv2d(_channels, config_channels(int(256 * ratio), '%s.branch1_1b.conv.weight' % prefix), kernel_size=(3, 1), stride=1, padding=(1, 0), bn=bn)
        # branch2
        config_channels.channels = channels
        self.branch2_0 = Conv2d(config_channels.channels, config_channels(int(384 * ratio), '%s.branch2_0.conv.weight' % prefix), kernel_size=1, stride=1, bn=bn)
        self.branch2_1 = Conv2d(config_channels.channels, config_channels(int(448 * ratio), '%s.branch2_1.conv.weight' % prefix), kernel_size=(3, 1), stride=1, padding=(1, 0), bn=bn)
        self.branch2_2 = Conv2d(config_channels.channels, config_channels(int(512 * ratio), '%s.branch2_2.conv.weight' % prefix), kernel_size=(1, 3), stride=1, padding=(0, 1), bn=bn)
        _channels = config_channels.channels
        self.branch2_3a = Conv2d(_channels, config_channels(int(256 * ratio), '%s.branch2_3a.conv.weight' % prefix), kernel_size=(1, 3), stride=1, padding=(0, 1), bn=bn)
        self.branch2_3b = Conv2d(_channels, config_channels(int(256 * ratio), '%s.branch2_3b.conv.weight' % prefix), kernel_size=(3, 1), stride=1, padding=(1, 0), bn=bn)
        # branch3
        config_channels.channels = channels
        branch = []
        branch.append(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False))
        branch.append(Conv2d(config_channels.channels, int(256 * ratio), kernel_size=1, stride=1, bn=bn))
        self.branch3 = nn.Sequential(*branch)
        # output
        config_channels.channels = self.branch0.conv.weight.size(0) + self.branch1_1a.conv.weight.size(0) + self.branch1_1b.conv.weight.size(0) + self.branch2_3a.conv.weight.size(0) + self.branch2_3b.conv.weight.size(0) + self.branch3[-1].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception4(nn.Module):
    def __init__(self, config_channels, anchors, num_cls, ratio=1):
        nn.Module.__init__(self)
        features = []
        bn = config_channels.config.getboolean('batch_norm', 'enable')
        features.append(Conv2d(config_channels.channels, config_channels(32, 'features.%d.conv.weight' % len(features)), kernel_size=3, stride=2, bn=bn))
        features.append(Conv2d(config_channels.channels, config_channels(32, 'features.%d.conv.weight' % len(features)), kernel_size=3, stride=1, bn=bn))
        features.append(Conv2d(config_channels.channels, config_channels(64, 'features.%d.conv.weight' % len(features)), kernel_size=3, stride=1, padding=1, bn=bn))
        features.append(Mixed_3a(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Mixed_4a(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Mixed_5a(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Reduction_A(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio)) # Mixed_6a
        features.append(Inception_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Reduction_B(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio)) # Mixed_7a
        features.append(Inception_C(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_C(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(Inception_C(config_channels, 'features.%d' % len(features), bn=bn, ratio=ratio))
        features.append(nn.Conv2d(config_channels.channels, model.output_channels(len(anchors), num_cls), 1))
        self.features = nn.Sequential(*features)
        self.init(config_channels)

    def init(self, config_channels):
        try:
            gamma = config_channels.config.getboolean('batch_norm', 'gamma')
        except (configparser.NoSectionError, configparser.NoOptionError):
            gamma = True
        try:
            beta = config_channels.config.getboolean('batch_norm', 'beta')
        except (configparser.NoSectionError, configparser.NoOptionError):
            beta = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.weight.requires_grad = gamma
                m.bias.requires_grad = beta
        try:
            if config_channels.config.getboolean('model', 'pretrained'):
                settings = pretrained_settings['inceptionv4'][config_channels.config.get('inception4', 'pretrained')]
                logging.info('use pretrained model: ' + str(settings))
                state_dict = self.state_dict()
                for key, value in torch.utils.model_zoo.load_url(settings['url']).items():
                    if key in state_dict:
                        state_dict[key] = value
                self.load_state_dict(state_dict)
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass

    def forward(self, x):
        return self.features(x)

    def scope(self, name):
        return '.'.join(name.split('.')[:-2])
