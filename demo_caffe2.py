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
import logging
import logging.config
import hashlib

import yaml
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
import cv2

import utils
import transform


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    model_dir = utils.get_model_dir(config)
    height, width = tuple(map(int, config.get('image', 'size').split()))
    resize = transform.parse_transform(config, config.get('transform', 'resize_test'))
    transform_image = transform.get_transform(config, config.get('transform', 'image_test').split())
    transform_tensor = transform.get_transform(config, config.get('transform', 'tensor').split())
    # load image
    image_bgr = cv2.imread('image.jpg')
    image_resized = resize(image_bgr, height, width)
    image = transform_image(image_resized)
    tensor = transform_tensor(image).unsqueeze(0)
    # Caffe2
    init_net = caffe2_pb2.NetDef()
    with open(os.path.join(model_dir, 'init_net.pb'), 'rb') as f:
        init_net.ParseFromString(f.read())
    predict_net = caffe2_pb2.NetDef()
    with open(os.path.join(model_dir, 'predict_net.pb'), 'rb') as f:
        predict_net.ParseFromString(f.read())
    p = workspace.Predictor(init_net, predict_net)
    results = p.run([tensor.numpy()])
    logging.info(utils.abs_mean(results[0]))
    logging.info(hashlib.md5(results[0].tostring()).hexdigest())


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
