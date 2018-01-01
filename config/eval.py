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

import numpy
import humanize
import pybenchmark


def timestamp(env, **kwargs):
    return float(env.now.timestamp())


def time(env, **kwargs):
    return env.now.strftime('%Y-%m-%d %H:%M:%S')


def step(env, **kwargs):
    return env.step


def epoch(env, **kwargs):
    return env.epoch


def model(env, **kwargs):
    return env.config.get('model', 'dnn')


def size_dnn(env, **kwargs):
    return humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in env.inference.state_dict().values()))


def time_inference(env, **kwargs):
    return pybenchmark.stats['inference']['time']


def root(env, **kwargs):
    return os.path.basename(env.config.get('config', 'root'))


def cache_name(env, **kwargs):
    return env.config.get('cache', 'name')


def model_name(env, **kwargs):
    return env.config.get('model', 'name')


def category(env, **kwargs):
    return env.config.get('cache', 'category')


def dataset_size(env, **kwargs):
    return len(env.loader.dataset)


def detect_threshold(env, **kwargs):
    return env.config.getfloat('detect', 'threshold')


def detect_overlap(env, **kwargs):
    return env.config.getfloat('detect', 'overlap')


def eval_iou(env, **kwargs):
    return env.config.getfloat('eval', 'iou')


def eval_mean_ap(env, **kwargs):
    return numpy.mean(list(kwargs['cls_ap'].values()))


def eval_ap(env, **kwargs):
    cls_ap = kwargs['cls_ap']
    return ', '.join(['%s=%f' % (env.category[c], cls_ap[c]) for c in sorted(cls_ap.keys())])


def hparam(env, **kwargs):
    return ', '.join([option + '=' + value for option, value in env._config.items('hparam')]) if hasattr(env, '_config') else None
