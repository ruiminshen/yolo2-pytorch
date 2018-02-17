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
import re

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet as _model
from torchvision.models.resnet import conv3x3

import model


class BasicBlock(nn.Module):
    def __init__(self, config_channels, prefix, channels, stride=1):
        nn.Module.__init__(self)
        channels_in = config_channels.channels
        self.conv1 = conv3x3(config_channels.channels, config_channels(channels, '%s.conv1.weight' % prefix), stride)
        self.bn1 = nn.BatchNorm2d(config_channels.channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(config_channels.channels, config_channels(channels, '%s.conv2.weight' % prefix))
        self.bn2 = nn.BatchNorm2d(config_channels.channels)
        if stride > 1 or channels_in != config_channels.channels:
            downsample = []
            downsample.append(nn.Conv2d(channels_in, config_channels.channels, kernel_size=1, stride=stride, bias=False))
            downsample.append(nn.BatchNorm2d(config_channels.channels))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, config_channels, prefix, channels, stride=1):
        nn.Module.__init__(self)
        channels_in = config_channels.channels
        self.conv1 = nn.Conv2d(config_channels.channels, config_channels(channels, '%s.conv1.weight' % prefix), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(config_channels.channels)
        self.conv2 = nn.Conv2d(config_channels.channels, config_channels(channels, '%s.conv2.weight' % prefix), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(config_channels.channels)
        self.conv3 = nn.Conv2d(config_channels.channels, config_channels(channels * 4, '%s.conv3.weight' % prefix), kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(config_channels.channels)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or channels_in != config_channels.channels:
            downsample = []
            downsample.append(nn.Conv2d(channels_in, config_channels.channels, kernel_size=1, stride=stride, bias=False))
            downsample.append(nn.BatchNorm2d(config_channels.channels))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(_model.ResNet):
    def __init__(self, config_channels, anchors, num_cls, block, layers):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(config_channels.channels, config_channels(64, 'conv1.weight'), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(config_channels.channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(config_channels, 'layer1', block, 64, layers[0])
        self.layer2 = self._make_layer(config_channels, 'layer2', block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(config_channels, 'layer3', block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(config_channels, 'layer4', block, 512, layers[3], stride=2)
        self.conv = nn.Conv2d(config_channels.channels, model.output_channels(len(anchors), num_cls), 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, config_channels, prefix, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(config_channels, '%s.%d' % (prefix, len(layers)), channels, stride))
        for i in range(1, blocks):
            layers.append(block(config_channels, '%s.%d' % (prefix, len(layers)), channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return self.conv(x)

    def scope(self, name):
        comp = name.split('.')[:-1]
        try:
            comp[-1] = re.search('[(conv)|(bn)](\d+)', comp[-1]).group(1)
        except AttributeError:
            if len(comp) > 1:
                if comp[-2] == 'downsample':
                    comp = comp[:-1]
                else:
                    assert False, name
            else:
                assert comp[-1] == 'conv', name
        return '.'.join(comp)


def resnet18(config_channels, anchors, num_cls, **kwargs):
    model = ResNet(config_channels, anchors, num_cls, BasicBlock, [2, 2, 2, 2], **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = _model.model_urls['resnet18']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def resnet34(config_channels, anchors, num_cls, **kwargs):
    model = ResNet(config_channels, anchors, num_cls, BasicBlock, [3, 4, 6, 3], **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = _model.model_urls['resnet34']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def resnet50(config_channels, anchors, num_cls, **kwargs):
    model = ResNet(config_channels, anchors, num_cls, Bottleneck, [3, 4, 6, 3], **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = _model.model_urls['resnet50']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def resnet101(config_channels, anchors, num_cls, **kwargs):
    model = ResNet(config_channels, anchors, num_cls, Bottleneck, [3, 4, 23, 3], **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = _model.model_urls['resnet101']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def resnet152(config_channels, anchors, num_cls, **kwargs):
    model = ResNet(config_channels, anchors, num_cls, Bottleneck, [3, 8, 36, 3], **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = _model.model_urls['resnet152']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model
