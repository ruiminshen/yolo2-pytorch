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

import os
import configparser

import numpy as np
import humanize
import pybenchmark


class Timestamp(object):
    def __call__(self, env, **kwargs):
        return float(env.now.timestamp())


class Time(object):
    def __call__(self, env, **kwargs):
        return env.now.strftime('%Y-%m-%d %H:%M:%S')

    def get_format(self, workbook, worksheet):
        return workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})


class Step(object):
    def __call__(self, env, **kwargs):
        return env.step


class Epoch(object):
    def __call__(self, env, **kwargs):
        return env.epoch


class Model(object):
    def __call__(self, env, **kwargs):
        return env.config.get('model', 'dnn')


class SizeDnn(object):
    def __call__(self, env, **kwargs):
        return sum(var.cpu().numpy().nbytes for var in env.inference.state_dict().values())

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class SizeDnnNature(object):
    def __call__(self, env, **kwargs):
        return humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in env.inference.state_dict().values()))


class TimeInference(object):
    def __call__(self, env, **kwargs):
        return pybenchmark.stats['inference']['time']

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class Root(object):
    def __call__(self, env, **kwargs):
        return os.path.basename(env.config.get('config', 'root'))


class CacheName(object):
    def __call__(self, env, **kwargs):
        return env.config.get('cache', 'name')


class ModelName(object):
    def __call__(self, env, **kwargs):
        return env.config.get('model', 'name')


class Category(object):
    def __call__(self, env, **kwargs):
        return env.config.get('cache', 'category')


class DatasetSize(object):
    def __call__(self, env, **kwargs):
        return len(env.loader.dataset)

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class DetectThreshold(object):
    def __call__(self, env, **kwargs):
        return env.config.getfloat('detect', 'threshold')

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class DetectThresholdCls(object):
    def __call__(self, env, **kwargs):
        return env.config.getfloat('detect', 'threshold_cls')

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class DetectFix(object):
    def __call__(self, env, **kwargs):
        return env.config.getboolean('detect', 'fix')

    def format(self, workbook, worksheet, num, col):
        format_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        format_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'cell', 'criteria': '==', 'value': '1', 'format': format_green})
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'cell', 'criteria': '<>', 'value': '1', 'format': format_red})


class DetectOverlap(object):
    def __call__(self, env, **kwargs):
        return env.config.getfloat('detect', 'overlap')

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class EvalIou(object):
    def __call__(self, env, **kwargs):
        return env.config.getfloat('eval', 'iou')

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class EvalMeanAp(object):
    def __call__(self, env, **kwargs):
        return np.mean(list(kwargs['cls_ap'].values()))

    def format(self, workbook, worksheet, num, col):
        worksheet.conditional_format(1, col, num + 1, col, {'type': 'data_bar', 'bar_color': '#FFC7CE'})


class EvalAp(object):
    def __call__(self, env, **kwargs):
        cls_ap = kwargs['cls_ap']
        return ', '.join(['%s=%f' % (env.category[c], cls_ap[c]) for c in sorted(cls_ap.keys())])


class Hparam(object):
    def __call__(self, env, **kwargs):
        try:
            return ', '.join([option + '=' + value for option, value in env._config.items('hparam')])
        except AttributeError:
            return None


class Optimizer(object):
    def __call__(self, env, **kwargs):
        try:
            return env._config.get('train', 'optimizer')
        except (AttributeError, configparser.NoOptionError):
            return None


class Scheduler(object):
    def __call__(self, env, **kwargs):
        try:
            return env._config.get('train', 'scheduler')
        except (AttributeError, configparser.NoOptionError):
            return None
