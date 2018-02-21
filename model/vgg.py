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

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models.vgg as _model
from torchvision.models.vgg import model_urls, cfg

import model


class VGG(_model.VGG):
    def __init__(self, config_channels, anchors, num_cls, features):
        nn.Module.__init__(self)
        self.features = features
        self.conv = nn.Conv2d(config_channels.channels, model.output_channels(len(anchors), num_cls), 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return self.conv(x)


def make_layers(config_channels, cfg, batch_norm=False):
    features = []
    for v in cfg:
        if v == 'M':
            features += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(config_channels.channels, config_channels(v, 'features.%d.weight' % len(features)), kernel_size=3, padding=1)
            if batch_norm:
                features += [conv2d, nn.BatchNorm2d(config_channels.channels), nn.ReLU(inplace=True)]
            else:
                features += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*features)


def vgg11(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['A']))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg11']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def vgg11_bn(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['A'], batch_norm=True))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg11_bn']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def vgg13(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['B']))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg13']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def vgg13_bn(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['B'], batch_norm=True))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg13_bn']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def vgg16(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['D']))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg16']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def vgg16_bn(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['D'], batch_norm=True))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg16_bn']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def vgg19(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['E']))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg19']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def vgg19_bn(config_channels, anchors, num_cls):
    model = VGG(config_channels, anchors, num_cls, make_layers(config_channels, cfg['E'], batch_norm=True))
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['vgg19_bn']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model
