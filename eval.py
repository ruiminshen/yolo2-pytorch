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
import argparse
import configparser
import datetime
import json
import logging
import logging.config
import multiprocessing
import importlib
import inspect
import inflection
import hashlib
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import tqdm
import humanize
import pybenchmark
import tinydb
import xlsxwriter
import cv2

import transform
import model
import utils.data
import utils.iou.torch
import utils.postprocess
import utils.train
import utils.visualize
from detect import get_logits, postprocess


def _matching(positive, index):
    detected = set()
    tp = np.zeros([len(positive)], np.bool)
    for i, (positive, index) in enumerate(zip(positive, index)):
        if positive and index not in detected:
            tp[i] = True
            detected.add(index)
    return tp


def matching(data_yx_min, data_yx_max, yx_min, yx_max, threshold):
    if data_yx_min.numel() > 0:
        matrix = utils.iou.torch.iou_matrix(yx_min, yx_max, data_yx_min, data_yx_max)
        iou, index = torch.max(matrix, -1)
        positive = iou > threshold
        tp = _matching(positive.cpu().numpy(), index.cpu().numpy())
    else:
        tp = np.zeros([yx_min.size(0)], np.bool)
    return tp


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def average_precision(config, tp, num, dtype=np.float):
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if num > 0:
        rec = tp / num
    else:
        rec = np.zeros(len(tp), dtype=dtype)
    prec = tp / np.maximum(tp + fp, np.finfo(dtype).eps)
    return voc_ap(rec, prec, config.getboolean('eval', 'metric07'))


def norm_bbox_data(data, keys='yx_min, yx_max'.split(', ')):
    height, width = data['image'].size()[1:3]
    scale = utils.ensure_device(torch.from_numpy(np.reshape(np.array([height, width], dtype=np.float32), [1, 1, 2])))
    for key in keys:
        data[key] = data[key] / scale
    return keys


def norm_bbox_pred(pred, keys='yx_min, yx_max'.split(', ')):
    rows, cols = pred['feature'].size()[-2:]
    scale = utils.ensure_device(torch.from_numpy(np.reshape(np.array([rows, cols], dtype=np.float32), [1, 1, 1, 2])))
    for key in keys:
        pred[key] = pred[key] / scale
    return keys


def filter_valid(yx_min, yx_max, cls, difficult):
    mask = torch.prod(yx_min < yx_max, -1) & (difficult < 1)
    _mask = torch.unsqueeze(mask, -1).repeat(1, 2) # PyTorch's bug
    cls, = (t[mask] for t in (cls,))
    yx_min, yx_max = (t[_mask].view(-1, 2) for t in (yx_min, yx_max))
    return yx_min, yx_max, cls


def filter_cls_data(yx_min, yx_max, mask):
    if mask.numel() > 0:
        _mask = torch.unsqueeze(mask, -1).repeat(1, 2)  # PyTorch's bug
        yx_min, yx_max = (t[_mask].view(-1, 2) for t in (yx_min, yx_max))
    else:  # all bboxes are difficult
        yx_min = utils.ensure_device(torch.zeros(0, 2))
        yx_max = utils.ensure_device(torch.zeros(0, 2))
    return yx_min, yx_max


def filter_cls_pred(yx_min, yx_max, score, mask):
    _mask = torch.unsqueeze(mask, -1).repeat(1, 2)  # PyTorch's bug
    yx_min, yx_max = (t[_mask].view(-1, 2) for t in (yx_min, yx_max))
    score = score[mask]
    return yx_min, yx_max, score


class Eval(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.model_dir = utils.get_model_dir(config)
        self.cache_dir = utils.get_cache_dir(config)
        self.category = utils.get_category(config, self.cache_dir)
        self.draw_bbox = utils.visualize.DrawBBox(self.category)
        self.loader = self.get_loader()
        self.anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
        self.path, self.step, self.epoch = utils.train.load_model(self.model_dir)
        state_dict = torch.load(self.path, map_location=lambda storage, loc: storage)
        dnn = utils.parse_attr(config.get('model', 'dnn'))(model.ConfigChannels(config, state_dict), self.anchors, len(self.category))
        dnn.load_state_dict(state_dict)
        logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in dnn.state_dict().values())))
        self.inference = model.Inference(config, dnn, self.anchors)
        self.inference.eval()
        if torch.cuda.is_available():
            self.inference.cuda()
        path = self.model_dir + '.ini'
        if os.path.exists(path):
            self._config = configparser.ConfigParser()
            self._config.read(path)
        else:
            logging.warning('training config (%s) not found' % path)
        self.now = datetime.datetime.now()
        self.mapper = dict([(inflection.underscore(name), member()) for name, member in inspect.getmembers(importlib.machinery.SourceFileLoader('', self.config.get('eval', 'mapper')).load_module()) if inspect.isclass(member)])

    def get_loader(self):
        paths = [os.path.join(self.cache_dir, phase + '.pkl') for phase in self.config.get('eval', 'phase').split()]
        dataset = utils.data.Dataset(utils.data.load_pickles(paths))
        logging.info('num_examples=%d' % len(dataset))
        size = tuple(map(int, self.config.get('image', 'size').split()))
        try:
            workers = self.config.getint('data', 'workers')
        except configparser.NoOptionError:
            workers = multiprocessing.cpu_count()
        collate_fn = utils.data.Collate(
            transform.parse_transform(self.config, self.config.get('transform', 'resize_eval')),
            [size],
            transform_image=transform.get_transform(self.config, self.config.get('transform', 'image_test').split()),
            transform_tensor=transform.get_transform(self.config, self.config.get('transform', 'tensor').split()),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, num_workers=workers, collate_fn=collate_fn)

    def filter_cls(self, c, path, data_yx_min, data_yx_max, data_cls, yx_min, yx_max, cls, score):
        data_yx_min, data_yx_max = filter_cls_data(data_yx_min, data_yx_max, data_cls == c)
        yx_min, yx_max, score = filter_cls_pred(yx_min, yx_max, score, cls == c)
        tp = pybenchmark.profile('matching')(matching)(data_yx_min, data_yx_max, yx_min, yx_max, self.config.getfloat('eval', 'iou'))
        if self.config.getboolean('eval', 'debug'):
            self.debug_visualize(data_yx_min, data_yx_max, yx_min, yx_max, c, tp, path)
        return score, tp

    def debug_data(self, data):
        for i, t in enumerate(torch.unbind(data['image'])):
            a = t.cpu().numpy()
            logging.info('image%d: %f %s' % (i, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()))
        for i, t in enumerate(torch.unbind(data['tensor'])):
            a = t.cpu().numpy()
            logging.info('tensor%d: %f %s' % (i, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()))

    def debug_pred(self, pred):
        for i, t in enumerate(torch.unbind(pred['iou'])):
            a = t.cpu().numpy()
            logging.info('iou%d: %f %s' % (i, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()))
        for i, t in enumerate(torch.unbind(pred['center_offset'])):
            a = t.cpu().numpy()
            logging.info('center_offset%d: %f %s' % (i, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()))
        for i, t in enumerate(torch.unbind(pred['size_norm'])):
            a = t.cpu().numpy()
            logging.info('size_norm%d: %f %s' % (i, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()))
        for i, t in enumerate(torch.unbind(pred['logits'])):
            a = t.cpu().numpy()
            logging.info('logits%d: %f %s' % (i, utils.abs_mean(a), hashlib.md5(a.tostring()).hexdigest()))

    def debug_visualize(self, data_yx_min, data_yx_max, yx_min, yx_max, c, tp, path):
        canvas = cv2.imread(path)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        size = np.reshape(np.array(canvas.shape[:2], np.float32), [1, 2])
        data_yx_min, data_yx_max, yx_min, yx_max = (np.reshape(t.cpu().numpy(), [-1, 2]) * size for t in (data_yx_min, data_yx_max, yx_min, yx_max))
        canvas = self.draw_bbox(canvas, data_yx_min, data_yx_max, colors=['g'])
        canvas = self.draw_bbox(canvas, *(a[tp] for a in (yx_min, yx_max)), colors=['w'])
        fp = ~tp
        canvas = self.draw_bbox(canvas, *(a[fp] for a in (yx_min, yx_max)), colors=['k'])
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(canvas)
        ax.set_title('tp=%d, fp=%d' % (np.sum(tp), np.sum(fp)))
        fig.canvas.set_window_title(self.category[c] + ': ' + path)
        plt.show()
        plt.close(fig)

    def stat_ap(self):
        cls_num = [0 for _ in self.category]
        cls_score = [np.array([], dtype=np.float32) for _ in self.category]
        cls_tp = [np.array([], dtype=np.bool) for _ in self.category]
        for data in tqdm.tqdm(self.loader):
            for key in data:
                t = data[key]
                if torch.is_tensor(t):
                    data[key] = utils.ensure_device(t)
            tensor = torch.autograd.Variable(data['tensor'], volatile=True)
            pred = pybenchmark.profile('inference')(model._inference)(self.inference, tensor)
            pred['iou'] = pred['iou'].contiguous()
            logits = get_logits(pred)
            pred['prob'] = F.softmax(logits, -1)
            for key in pred:
                pred[key] = pred[key].data
            if self.config.getboolean('eval', 'debug'):
                self.debug_data(data)
                self.debug_pred(pred)
            norm_bbox_data(data)
            norm_bbox_pred(pred)
            for path, difficult, image, data_yx_min, data_yx_max, data_cls, iou, yx_min, yx_max, prob in zip(*(data[key] for key in 'path, difficult'.split(', ')), *(torch.unbind(data[key]) for key in 'image, yx_min, yx_max, cls'.split(', ')), *(torch.unbind(pred[key]) for key in 'iou, yx_min, yx_max, prob'.split(', '))):
                data_yx_min, data_yx_max, data_cls = filter_valid(data_yx_min, data_yx_max, data_cls, difficult)
                for c in data_cls.cpu().numpy():
                    cls_num[c] += 1
                iou = iou.view(-1)
                yx_min, yx_max, prob = (t.view(-1, t.size(-1)) for t in (yx_min, yx_max, prob))
                ret = postprocess(self.config, iou, yx_min, yx_max, prob)
                if ret is not None:
                    iou, yx_min, yx_max, cls, score = ret
                    for c in set(cls.cpu().numpy()):
                        c = int(c)  # PyTorch's bug
                        _score, tp = self.filter_cls(c, path, data_yx_min, data_yx_max, data_cls, yx_min, yx_max, cls, score)
                        cls_score[c] = np.append(cls_score[c], _score.cpu().numpy())
                        cls_tp[c] = np.append(cls_tp[c], tp)
        return cls_num, cls_score, cls_tp

    def merge_ap(self, cls_num, cls_score, cls_tp):
        cls_ap = {}
        for c, (num, score, tp) in enumerate(zip(cls_num, cls_score, cls_tp)):
            if num > 0:
                indices = np.argsort(-score)
                tp = tp[indices]
                cls_ap[c] = average_precision(self.config, tp, num)
        return cls_ap

    def save_db(self, cls_ap, path):
        with tinydb.TinyDB(path) as db:
            row = dict([(key, fn(self, cls_ap=cls_ap)) for key, fn in self.mapper.items()])
            db.insert(row)

    def save_xlsx(self, df, path, worksheet='worksheet'):
        with xlsxwriter.Workbook(path, {'strings_to_urls': False, 'nan_inf_to_errors': True}) as workbook:
            worksheet = workbook.add_worksheet(worksheet)
            for j, key in enumerate(df):
                worksheet.write(0, j, key)
                try:
                    m = self.mapper[key]
                except (KeyError, AttributeError):
                    m = None
                if hasattr(m, 'get_format'):
                    fmt = m.get_format(workbook, worksheet)
                else:
                    fmt = None
                for i, value in enumerate(df[key]):
                    worksheet.write(1 + i, j, value, fmt)
                if hasattr(m, 'format'):
                    m.format(workbook, worksheet, i, j)
            worksheet.autofilter(0, 0, i, len(self.mapper) - 1)
            worksheet.freeze_panes(1, 0)

    def logging(self, cls_ap):
        for c in cls_ap:
            logging.info('%s=%f' % (self.category[c], cls_ap[c]))
        logging.info(np.mean(list(cls_ap.values())))

    def __call__(self):
        cls_num, cls_score, cls_tp = self.stat_ap()
        cls_ap = self.merge_ap(cls_num, cls_score, cls_tp)
        path = utils.get_eval_db(self.config)
        self.save_db(cls_ap, path)
        with open(path, 'r') as f:
            df = pd.read_json(json.dumps(json.load(f)['_default']), orient='index', convert_dates=False)
        df = df[sorted(df)]
        try:
            df = df.sort_values(self.config.get('eval', 'sort'))
        except configparser.NoOptionError:
            pass
        self.save_xlsx(df, os.path.splitext(path)[0] + '.xlsx')
        self.logging(cls_ap)
        return cls_ap


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    eval = Eval(args, config)
    eval()
    logging.info(pybenchmark.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
