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

import onnx
import onnx_caffe2.backend
import onnx_caffe2.helper

import utils


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    if args.level:
        logging.getLogger().setLevel(args.level.upper())
    model_dir = utils.get_model_dir(config)
    model = onnx.load(model_dir + '.onnx')
    onnx.checker.check_model(model)
    init_net, predict_net = onnx_caffe2.backend.Caffe2Backend.onnx_graph_to_caffe2_net(model.graph, device='CPU')
    onnx_caffe2.helper.save_caffe2_net(init_net, os.path.join(model_dir, 'init_net.pb'))
    onnx_caffe2.helper.save_caffe2_net(predict_net, os.path.join(model_dir, 'predict_net.pb'), output_txt=True)
    logging.info(model_dir)
    if args.benchmark:
        benchmark = onnx_caffe2.helper.benchmark_caffe2_model(init_net, predict_net)
        print('benchmark=%f(milliseconds)' % benchmark)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--benchmark', action='store_true')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    main()
