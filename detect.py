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


def conv_logits(pred):
    if 'logits' in pred:
        logits = pred['logits'].contiguous()
        prob, cls = torch.max(F.softmax(logits, -1), -1)
    else:
        size = pred['iou'].size()
        prob = torch.autograd.Variable(utils.ensure_device(torch.ones(*size)))
        cls = torch.autograd.Variable(utils.ensure_device(torch.zeros(*size).int()))
    return prob, cls


class Detect(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.cache_dir = utils.get_cache_dir(config)
        self.model_dir = utils.get_model_dir(config)
        self.category = utils.get_category(config, self.cache_dir if os.path.exists(self.cache_dir) else None)
        self.draw_bbox = utils.visualize.DrawBBox(config, self.category)
        self.anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
        self.height, self.width = tuple(map(int, config.get('image', 'size').split()))
        self.dnn = utils.parse_attr(config.get('model', 'dnn'))(config, self.anchors, len(self.category))
        path, step, epoch = utils.train.load_model(self.model_dir)
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.dnn.load_state_dict(state_dict)
        self.inference = model.Inference(config, self.dnn, self.anchors)
        self.inference.eval()
        if torch.cuda.is_available():
            self.inference.cuda()
        logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in self.inference.state_dict().values())))
        self.create_cap()
        self.create_cap_size()
        self.create_writer()
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
        except:
            cap = os.path.expanduser(os.path.expandvars(self.args.input))
            assert os.path.exists(cap)
        self.cap = cv2.VideoCapture(cap)

    def create_cap_size(self):
        cap_height, cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self.args.crop:
            crop_ymin, crop_ymax, crop_xmin, crop_xmax = self.args.crop
            if crop_ymin <= 1 and crop_ymax <= 1 and crop_xmin <= 1 and crop_xmax <= 1:
                crop_ymin, crop_ymax = crop_ymin * cap_height, crop_ymax * cap_height
                crop_xmin, crop_xmax = crop_xmin * cap_width, crop_xmax * cap_width
            crop_ymin, crop_ymax, crop_xmin, crop_xmax = int(crop_ymin), int(crop_ymax), int(crop_xmin), int(crop_xmax)
            cap_height, cap_width = crop_ymax - crop_ymin, crop_xmax - crop_xmin
            self.crop_ymin, self.crop_ymax, self.crop_xmin, self.crop_xmax = crop_ymin, crop_ymax, crop_xmin, crop_xmax
        logging.info('cap_height, cap_width=%d, %d' % (cap_height, cap_width))
        self.cap_height, self.cap_width = cap_height, cap_width

    def create_writer(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logging.info('cap fps=%f' % fps)
        if self.args.output:
            path = os.path.expanduser(os.path.expandvars(self.args.output))
            if self.args.fourcc:
                fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc.upper())
            else:
                fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            return cv2.VideoWriter(path, fourcc, fps, (self.cap_width, self.cap_height))

    def get_image(self):
        ret, image_bgr = self.cap.read()
        if self.args.crop:
            image_bgr = image_bgr[self.crop_ymin:self.crop_ymax, self.crop_xmin:self.crop_xmax]
        return image_bgr

    def conv_tensor(self, image_bgr):
        image_resized = self.resize(image_bgr, self.height, self.width)
        image = self.transform_image(image_resized)
        tensor = self.transform_tensor(image)
        return utils.ensure_device(tensor.unsqueeze(0))

    def filter_visible(self, pred):
        try:
            pred['score'] = pred['iou']
            mask = pred['score'] > self.config.getfloat('detect', 'threshold')
        except configparser.NoOptionError:
            pred['score'] = pred['prob']
            mask = pred['score'] > self.config.getfloat('detect', 'threshold_cls')
        mask = mask.detach() # PyTorch's bug
        _mask = torch.unsqueeze(mask, -1)
        for key in 'yx_min, yx_max'.split(', '):
            pred[key] = pred[key][_mask].view(-1, 2)
        for key in 'cls, score'.split(', '):
            pred[key] = pred[key][mask]

    def __call__(self):
        image_bgr = self.get_image()
        tensor = self.conv_tensor(image_bgr)
        pred = pybenchmark.profile('inference')(model._inference)(self.inference, torch.autograd.Variable(tensor))
        rows, cols = pred['feature'].size()[-2:]
        _prob, pred['cls'] = conv_logits(pred)
        pred['prob'] = pred['iou'] * _prob
        self.filter_visible(pred)
        keep = pybenchmark.profile('nms')(utils.postprocess.nms)(*(pred[key].data for key in 'yx_min, yx_max, score'.split(', ')), self.config.getfloat('detect', 'overlap'))
        image_result = image_bgr.copy()
        if keep:
            yx_min, yx_max, cls = (pred[key].data.cpu().numpy()[keep] for key in 'yx_min, yx_max, cls'.split(', '))
            scale = np.array(image_result.shape[:2], np.float32) / [rows, cols]
            yx_min, yx_max = ((a * scale).astype(np.int) for a in (yx_min, yx_max))
            image_result = self.draw_bbox(image_result, yx_min, yx_max, cls)
        cv2.imshow('detection', image_result)
        if self.args.output:
            self.writer.write(image_result)
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
    parser.add_argument('-i', '--input', default=0)
    parser.add_argument('-k', '--keys', nargs='+', type=int, default=[ord(' ')], help='keys to dump images')
    parser.add_argument('-o', '--output', help='output video file')
    parser.add_argument('-f', '--format', default='%Y-%m-%d_%H-%M-%S.jpg', help='dump file name format')
    parser.add_argument('--crop', nargs='+', type=float, default=[], help='ymin ymax xmin xmax')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--fourcc', default='XVID', help='4-character code of codec used to compress the frames, such as XVID, MJPG')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
