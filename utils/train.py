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
import time
import operator
import logging

import torch


class Timer(object):
    def __init__(self, max, first=True):
        """
        A simple function object to determine time event.
        :author 申瑞珉 (Ruimin Shen)
        :param max: Number of seconds to trigger a time event.
        :param first: Should a time event to be triggered at the first time.
        """
        self.start = 0 if first else time.time()
        self.max = max

    def __call__(self):
        """
        Return a boolean value to indicate if the time event is occurred.
        :author 申瑞珉 (Ruimin Shen)
        """
        t = time.time()
        elapsed = t - self.start
        if elapsed > self.max:
            self.start = t
            return True
        else:
            return False


def load_model(model_dir, step=None, ext='.pth', ext_epoch='.epoch', map_location=lambda storage, loc: storage, logger=logging.info):
    """
    Load the latest checkpoint in a model directory.
    :author 申瑞珉 (Ruimin Shen)
    :param model_dir: The directory to store the model checkpoint files.
    :param step: If a integer value is given, the corresponding checkpoint will be loaded. Otherwise, the latest checkpoint (with the largest step value) will be loaded.
    :param ext: The extension of the model file.
    :param ext_epoch: The extension of the epoch file.
    :return:
    """
    if step is None:
        steps = [(int(n), n) for n, e in map(os.path.splitext, os.listdir(model_dir)) if n.isdigit() and e == ext]
        step, name = max(steps, key=operator.itemgetter(0))
    else:
        name = str(step)
    path = os.path.join(model_dir, name)
    if logger is not None:
        logger('load ' + path)
    try:
        with open(path + ext_epoch, 'r') as f:
            epoch = int(f.read())
    except (FileNotFoundError, ValueError):
        epoch = None
    _path = path + ext
    assert os.path.exists(_path), _path
    return _path, step, epoch


class Saver(object):
    def __init__(self, model_dir, keep, ext='.pth', ext_epoch='.epoch', logger=logging.info):
        """
        Manage several latest checkpoints (with the largest step values) in a model directory.
        :author 申瑞珉 (Ruimin Shen)
        :param model_dir: The directory to store the model checkpoint files.
        :param keep: How many latest checkpoints to be maintained.
        :param ext: The extension of the model file.
        :param ext_epoch: The extension of the epoch file.
        """
        self.model_dir = model_dir
        self.keep = keep
        self.ext = ext
        self.ext_epoch = ext_epoch
        self.logger = logger

    def __call__(self, obj, step, epoch=None):
        """
        Save the PyTorch module.
        :author 申瑞珉 (Ruimin Shen)
        :param obj: The PyTorch module to be saved.
        :param step: Current step.
        :param epoch: Current epoch.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, str(step))
        torch.save(obj, path + self.ext)
        if epoch is not None:
            with open(path + self.ext_epoch, 'w') as f:
                f.write(str(epoch))
        if self.logger is not None:
            self.logger('model saved into ' + path)
        self.tidy()
        return path

    def tidy(self):
        steps = [(int(n), n) for n, e in map(os.path.splitext, os.listdir(self.model_dir)) if n.isdigit() and e == self.ext]
        if len(steps) > self.keep:
            steps = sorted(steps, key=operator.itemgetter(0))
            remove = steps[:len(steps) - self.keep]
            for _, n in remove:
                path = os.path.join(self.model_dir, n)
                os.remove(path + self.ext)
                path_epoch = path + self.ext_epoch
                try:
                    os.remove(path_epoch)
                except FileNotFoundError:
                    self.logger(path_epoch + ' not found')
                logging.debug('tidy ' + path)


def load_sizes(config):
    sizes = [s.split(',') for s in config.get('data', 'sizes').split()]
    return [(int(height), int(width)) for height, width in sizes]
