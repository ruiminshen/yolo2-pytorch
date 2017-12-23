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

import onnx_caffe2.helper
import cv2

import utils
import transform


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    if args.level:
        logging.getLogger().setLevel(args.level.upper())
    model_dir = utils.get_model_dir(config)
    height, width = tuple(map(int, config.get('image', 'size').split()))
    resize = transform.parse_transform(config, config.get('transform', 'resize_test'))
    transform_image = transform.get_transform(config, config.get('transform', 'image_test').split())
    transform_tensor = transform.get_transform(config, config.get('transform', 'tensor').split())
    init_net = onnx_caffe2.helper.load_caffe2_net(os.path.join(model_dir, 'init_net.pb'))
    predict_net = onnx_caffe2.helper.load_caffe2_net(os.path.join(model_dir, 'predict_net.pb'))
    image_bgr = cv2.imread('image.jpg')
    image_resized = resize(image_bgr, height, width)
    image = transform_image(image_resized)
    tensor = transform_tensor(image).unsqueeze(0)
    _, results = onnx_caffe2.helper.c2_native_run_net(init_net, predict_net, tensor.numpy())
    print(results)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    main()
