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

import torch
import torch.nn as nn
from pretrainedmodels.models.inceptionv4 import InceptionV4, BasicConv2d, Mixed_3a, Mixed_4a, Mixed_5a, Inception_A, Reduction_A, Inception_B, Reduction_B, Inception_C, pretrained_settings

import model


class Inception4(nn.Module):
    def __init__(self, config_channels, anchors, num_cls):
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            nn.Conv2d(1536, model.output_channels(len(anchors), num_cls), 1),
        )

        gamma = config_channels.config.getboolean('batch_norm', 'gamma')
        beta = config_channels.config.getboolean('batch_norm', 'beta')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.weight.requires_grad = gamma
                m.bias.requires_grad = beta

        if config_channels.config.getboolean('model', 'pretrained'):
            settings = pretrained_settings['inceptionv4'][config_channels.config.get('inception4', 'pretrained')]
            logging.info('use pretrained model: ' + str(settings))
            state_dict = self.state_dict()
            for key, value in torch.utils.model_zoo.load_url(settings['url']).items():
                if key in state_dict:
                    state_dict[key] = value
            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.features(x)
