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
import multiprocessing

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybenchmark
import tinydb
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import tqdm
import cv2

import transform
import model
import utils.data
import utils.iou.torch
import utils.postprocess
import utils.train
import utils.visualize


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


def norm_bbox(data, pred, keys='yx_min, yx_max'):
    size, image = (data[key] for key in 'size, image'.split(', '))
    _size = size.float()
    height, width = image.size()[1:3]
    scale = _size.view(-1, 1, 2) / utils.ensure_device(torch.from_numpy(np.reshape(np.array([height, width], dtype=np.float32), [1, 1, 2])))
    for key in keys.split(', '):
        data[key] = data[key] * scale
    rows, cols = pred['feature'].size()[-2:]
    scale = _size.view(-1, 1, 1, 2) / utils.ensure_device(torch.from_numpy(np.reshape(np.array([rows, cols], dtype=np.float32), [1, 1, 1, 2])))
    for key in keys.split(', '):
        pred[key] = pred[key] * scale


def conv_logits(pred):
    if 'logits' in pred:
        logits = pred['logits'].contiguous()
        prob, cls = torch.max(F.softmax(logits, -1), -1)
    else:
        size = pred['iou'].size()
        prob = torch.autograd.Variable(utils.ensure_device(torch.ones(*size)))
        cls = torch.autograd.Variable(utils.ensure_device(torch.zeros(*size).int()))
    return prob, cls


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
        self.draw_bbox = utils.visualize.DrawBBox(config, self.category)
        self.loader = self.get_loader()
        self.anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
        dnn = utils.parse_attr(config.get('model', 'dnn'))(config, self.anchors, len(self.category))
        checkpoint, self.step, self.epoch = utils.train.load_model(self.model_dir)
        dnn.load_state_dict(checkpoint['dnn'])
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
        self.mapper = utils.load_functions(self.config.get('eval', 'mapper'))

    def get_loader(self):
        paths = [os.path.join(self.cache_dir, phase + '.pkl') for phase in self.config.get('eval', 'phase').split()]
        dataset = utils.data.Dataset(paths)
        logging.warning('num_examples=%d' % len(dataset))
        size = tuple(map(int, self.config.get('image', 'size').split()))
        try:
            workers = self.config.getint('data', 'workers')
        except configparser.NoOptionError:
            workers = multiprocessing.cpu_count()
        collate_fn = utils.data.Collate(
            [size],
            resize=transform.parse_transform(self.config, self.config.get('transform', 'resize_eval')),
            transform_image=transform.get_transform(self.config, self.config.get('transform', 'image_test').split()),
            transform_tensor=transform.get_transform(self.config, self.config.get('transform', 'tensor').split()),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, num_workers=workers, collate_fn=collate_fn)

    def filter_visible(self, yx_min, yx_max, iou, prob, cls):
        try:
            score = iou
            mask = score > self.config.getfloat('detect', 'threshold')
        except configparser.NoOptionError:
            score = prob
            mask = score > self.config.getfloat('detect', 'threshold_cls')
        _mask = torch.unsqueeze(mask, -1).repeat(1, 2)  # PyTorch's bug
        yx_min, yx_max = (t[_mask].view(-1, 2) for t in (yx_min, yx_max))
        cls, score = (t[mask].view(-1) for t in (cls, score))
        return yx_min, yx_max, cls, score

    def filter_cls(self, c, path, data_yx_min, data_yx_max, data_cls, yx_min, yx_max, cls, score):
        data_yx_min, data_yx_max = filter_cls_data(data_yx_min, data_yx_max, data_cls == c)
        yx_min, yx_max, score = filter_cls_pred(yx_min, yx_max, score, cls == c)
        tp = pybenchmark.profile('matching')(matching)(data_yx_min, data_yx_max, yx_min, yx_max, self.config.getfloat('eval', 'iou'))
        if self.config.getboolean('eval', 'debug'):
            self.debug(data_yx_min, data_yx_max, yx_min, yx_max, c, tp, path)
        return score, tp

    def debug(self, data_yx_min, data_yx_max, yx_min, yx_max, c, tp, path):
        canvas = cv2.imread(path)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = self.draw_bbox(canvas, *(t.cpu().numpy() for t in (data_yx_min, data_yx_max)), colors=['g'])
        canvas = self.draw_bbox(canvas, *(t.cpu().numpy()[tp] for t in (yx_min, yx_max)), colors=['w'])
        fp = ~tp
        canvas = self.draw_bbox(canvas, *(t.cpu().numpy()[fp] for t in (yx_min, yx_max)), colors=['k'])
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
            tensor = torch.autograd.Variable(data['tensor'])
            batch_size = tensor.size(0)
            pred = pybenchmark.profile('inference')(model._inference)(self.inference, tensor)
            _prob, pred['cls'] = conv_logits(pred)
            pred['iou'] = pred['iou'].contiguous()
            pred['prob'] = pred['iou'] * _prob
            for key in pred:
                pred[key] = pred[key].data
            norm_bbox(data, pred)
            for path, size, difficult, image, data_yx_min, data_yx_max, data_cls, yx_min, yx_max, iou, prob, cls in zip(*(data[key] for key in 'path, size, difficult'.split(', ')), *(torch.unbind(data[key]) for key in 'image, yx_min, yx_max, cls'.split(', ')), *(torch.unbind(pred[key].view(batch_size, -1, 2)) for key in 'yx_min, yx_max'.split(', ')), *(torch.unbind(pred[key].view(batch_size, -1)) for key in 'iou, prob, cls'.split(', '))):
                data_yx_min, data_yx_max, data_cls = filter_valid(data_yx_min, data_yx_max, data_cls, difficult)
                for c in data_cls.cpu().numpy():
                    cls_num[c] += 1
                yx_min, yx_max, cls, score = self.filter_visible(yx_min, yx_max, iou, prob, cls)
                keep = pybenchmark.profile('nms')(utils.postprocess.nms)(yx_min, yx_max, score, self.config.getfloat('detect', 'overlap'))
                if keep:
                    keep = utils.ensure_device(torch.LongTensor(keep))
                    yx_min, yx_max, cls, score = (t[keep] for t in (yx_min, yx_max, cls, score))
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

    def save_db(self, cls_ap):
        path = utils.get_eval_db(self.config)
        with tinydb.TinyDB(path) as db:
            row = dict([(name, fn(self, cls_ap=cls_ap)) for name, fn in self.mapper])
            db.insert(row)

    def save_tsv(self):
        path = utils.get_eval_db(self.config)
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.read_json(json.dumps(data['_default']), orient='index')
        df = df[sorted(df)]
        path = os.path.splitext(path)[0] + '.tsv'
        df.to_csv(path, index=False, sep='\t')

    def logging(self, cls_ap):
        for c in cls_ap:
            logging.info('%s=%f' % (self.category[c], cls_ap[c]))
        logging.info(np.mean(list(cls_ap.values())))

    def __call__(self):
        cls_num, cls_score, cls_tp = self.stat_ap()
        cls_ap = self.merge_ap(cls_num, cls_score, cls_tp)
        self.save_db(cls_ap)
        self.save_tsv()
        self.logging(cls_ap)
        return cls_ap


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    if args.level:
        logging.getLogger().setLevel(args.level.upper())
    eval = Eval(args, config)
    eval()
    logging.info(pybenchmark.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    main()
