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
import re
import configparser
import importlib

import numpy as np
import pandas as pd
import torch.autograd


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, yx_min, yx_max, cls):
        for t in self.transforms:
            img, yx_min, yx_max, cls = t(img, yx_min, yx_max, cls)
        return img, yx_min, yx_max, cls


class RegexList(list):
    def __init__(self, l):
        for s in l:
            prog = re.compile(s)
            self.append(prog)

    def __call__(self, s):
        for prog in self:
            if prog.match(s):
                return True
        return False


def get_cache_dir(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    name = config.get('cache', 'name')
    return os.path.join(root, name)


def get_model_dir(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    name = config.get('model', 'name')
    model = config.get('model', 'dnn')
    return os.path.join(root, name, model)


def get_eval_db(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    db = config.get('eval', 'db')
    return os.path.join(root, db)


def get_category(config, cache_dir=None):
    path = os.path.expanduser(os.path.expandvars(config.get('cache', 'category'))) if cache_dir is None else os.path.join(cache_dir, 'category')
    with open(path, 'r') as f:
        return [line.strip() for line in f]


def get_anchors(config, dtype=np.float32):
    path = os.path.expanduser(os.path.expandvars(config.get('model', 'anchors')))
    df = pd.read_csv(path, sep='\t', dtype=dtype)
    return df[['height', 'width']].values


def parse_attr(s):
    m, n = s.rsplit('.', 1)
    m = importlib.import_module(m)
    return getattr(m, n)


def load_config(config, paths):
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        assert os.path.exists(path)
        config.read(path)


def modify_config(config, cmd):
    var, value = cmd.split('=')
    section, option = var.split('/')
    if value:
        config.set(section, option, value)
    else:
        try:
            config.remove_option(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass


def ensure_device(t, device_id=None, async=False):
    if torch.cuda.is_available():
        t = t.cuda(device_id, async)
    return t


def abs_mean(data, dtype=np.float32):
    assert isinstance(data, np.ndarray)
    return np.sum(np.abs(data)) / dtype(data.size)
