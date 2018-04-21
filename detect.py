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

import argparse
import configparser
import logging
import logging.config
import os
import time
import yaml

import numpy as np
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import humanize
import pybenchmark
import cv2

import transform
import model
import utils.postprocess
import utils.train
import utils.visualize


def get_logits(pred):
    if 'logits' in pred:
        return pred['logits'].contiguous()
    else:
        size = pred['iou'].size()
        return torch.autograd.Variable(utils.ensure_device(torch.ones(*size, 1)))


def filter_visible(config, iou, yx_min, yx_max, prob):
    prob_cls, cls = torch.max(prob, -1)
    if config.getboolean('detect', 'fix'):
        mask = (iou * prob_cls) > config.getfloat('detect', 'threshold_cls')
    else:
        mask = iou > config.getfloat('detect', 'threshold')
    iou, prob_cls, cls = (t[mask].view(-1) for t in (iou, prob_cls, cls))
    _mask = torch.unsqueeze(mask, -1).repeat(1, 2)  # PyTorch's bug
    yx_min, yx_max = (t[_mask].view(-1, 2) for t in (yx_min, yx_max))
    num = prob.size(-1)
    _mask = torch.unsqueeze(mask, -1).repeat(1, num)  # PyTorch's bug
    prob = prob[_mask].view(-1, num)
    return iou, yx_min, yx_max, prob, prob_cls, cls


def postprocess(config, iou, yx_min, yx_max, prob):
    iou, yx_min, yx_max, prob, prob_cls, cls = filter_visible(config, iou, yx_min, yx_max, prob)
    keep = pybenchmark.profile('nms')(utils.postprocess.nms)(iou, yx_min, yx_max, config.getfloat('detect', 'overlap'))
    if keep:
        keep = utils.ensure_device(torch.LongTensor(keep))
        iou, yx_min, yx_max, prob, prob_cls, cls = (t[keep] for t in (iou, yx_min, yx_max, prob, prob_cls, cls))
        if config.getboolean('detect', 'fix'):
            score = torch.unsqueeze(iou, -1) * prob
            mask = score > config.getfloat('detect', 'threshold_cls')
            indices, cls = torch.unbind(mask.nonzero(), -1)
            yx_min, yx_max = (t[indices] for t in (yx_min, yx_max))
            score = score[mask]
        else:
            score = iou
        return iou, yx_min, yx_max, cls, score


class Detect(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.cache_dir = utils.get_cache_dir(config)
        self.model_dir = utils.get_model_dir(config)
        self.category = utils.get_category(config, self.cache_dir if os.path.exists(self.cache_dir) else None)
        self.draw_bbox = utils.visualize.DrawBBox(self.category, colors=args.colors, thickness=args.thickness)
        self.anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
        self.height, self.width = tuple(map(int, config.get('image', 'size').split()))
        self.path, self.step, self.epoch = utils.train.load_model(self.model_dir)
        state_dict = torch.load(self.path, map_location=lambda storage, loc: storage)
        self.dnn = utils.parse_attr(config.get('model', 'dnn'))(model.ConfigChannels(config, state_dict), self.anchors, len(self.category))
        self.dnn.load_state_dict(state_dict)
        self.inference = model.Inference(config, self.dnn, self.anchors)
        self.inference.eval()
        if torch.cuda.is_available():
            self.inference.cuda()
        logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in self.inference.state_dict().values())))
        self.cap = self.create_cap()
        self.keys = set(args.keys)
        self.resize = transform.parse_transform(config, config.get('transform', 'resize_test'))
        self.transform_image = transform.get_transform(config, config.get('transform', 'image_test').split())
        self.transform_tensor = transform.get_transform(config, config.get('transform', 'tensor').split())

    def __del__(self):
        cv2.destroyAllWindows()
        try:
            self.writer.release()
        except AttributeError:
            pass
        self.cap.release()

    def create_cap(self):
        try:
            cap = int(self.args.input)
        except ValueError:
            cap = os.path.expanduser(os.path.expandvars(self.args.input))
            assert os.path.exists(cap)
        return cv2.VideoCapture(cap)

    def create_writer(self, height, width):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logging.info('cap fps=%f' % fps)
        path = os.path.expanduser(os.path.expandvars(self.args.output))
        if self.args.fourcc:
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc.upper())
        else:
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return cv2.VideoWriter(path, fourcc, fps, (width, height))

    def get_image(self):
        ret, image_bgr = self.cap.read()
        if self.args.crop:
            image_bgr = image_bgr[self.crop_ymin:self.crop_ymax, self.crop_xmin:self.crop_xmax]
        return image_bgr

    def __call__(self):
        image_bgr = self.get_image()
        image_resized = self.resize(image_bgr, self.height, self.width)
        image = self.transform_image(image_resized)
        tensor = self.transform_tensor(image)
        tensor = utils.ensure_device(tensor.unsqueeze(0))
        pred = pybenchmark.profile('inference')(model._inference)(self.inference, torch.autograd.Variable(tensor, volatile=True))
        rows, cols = pred['feature'].size()[-2:]
        iou = pred['iou'].data.contiguous().view(-1)
        yx_min, yx_max = (pred[key].data.view(-1, 2) for key in 'yx_min, yx_max'.split(', '))
        logits = get_logits(pred)
        prob = F.softmax(logits, -1).data.view(-1, logits.size(-1))
        ret = postprocess(self.config, iou, yx_min, yx_max, prob)
        image_result = image_bgr.copy()
        if ret is not None:
            iou, yx_min, yx_max, cls, score = ret
            try:
                scale = self.scale
            except AttributeError:
                scale = utils.ensure_device(torch.from_numpy(np.array(image_result.shape[:2], np.float32) / np.array([rows, cols], np.float32)))
                self.scale = scale
            yx_min, yx_max = ((t * scale).cpu().numpy().astype(np.int) for t in (yx_min, yx_max))
            image_result = self.draw_bbox(image_result, yx_min, yx_max, cls)
        if self.args.output:
            if not hasattr(self, 'writer'):
                self.writer = self.create_writer(*image_result.shape[:2])
            self.writer.write(image_result)
        else:
            cv2.imshow('detection', image_result)
        if cv2.waitKey(0 if self.args.pause else 1) in self.keys:
            root = os.path.join(self.model_dir, 'snapshot')
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, time.strftime(self.args.format))
            cv2.imwrite(path, image_bgr)
            logging.warning('image dumped into ' + path)


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    detect = Detect(args, config)
    try:
        while detect.cap.isOpened():
            detect()
    except KeyboardInterrupt:
        logging.warning('interrupted')
    finally:
        logging.info(pybenchmark.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-i', '--input', default=-1)
    parser.add_argument('-k', '--keys', nargs='+', type=int, default=[ord(' ')], help='keys to dump images')
    parser.add_argument('-o', '--output', help='output video file')
    parser.add_argument('-f', '--format', default='%Y-%m-%d_%H-%M-%S.jpg', help='dump file name format')
    parser.add_argument('--crop', nargs='+', type=float, default=[], help='ymin ymax xmin xmax')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--fourcc', default='XVID', help='4-character code of codec used to compress the frames, such as XVID, MJPG')
    parser.add_argument('--thickness', default=3, type=int)
    parser.add_argument('--colors', nargs='+', default=[])
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
