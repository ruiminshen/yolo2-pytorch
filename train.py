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

import sys
import argparse
import configparser
import logging
import logging.config
import collections
import multiprocessing
import os
import shutil
import io
import hashlib
import subprocess
import pickle
import traceback
import yaml

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms
import tqdm
import humanize
import pybenchmark
import filelock
from tensorboardX import SummaryWriter

import model
import transform.augmentation
import utils.data
import utils.postprocess
import utils.train
import utils.visualize
import eval as _eval


def norm_data(data, height, width, rows, cols, keys='yx_min, yx_max'):
    _data = {key: data[key] for key in data}
    scale = utils.ensure_device(torch.from_numpy(np.reshape(np.array([rows / height, cols / width], dtype=np.float32), [1, 1, 2])))
    for key in keys.split(', '):
        _data[key] = _data[key] * scale
    return _data


def ensure_model(model):
    if torch.cuda.is_available():
        model.cuda()
        if torch.cuda.device_count() > 1:
            logging.info('%d GPUs are used' % torch.cuda.device_count())
            model = nn.DataParallel(model).cuda()
    return model


class SummaryWorker(multiprocessing.Process):
    def __init__(self, env):
        super(SummaryWorker, self).__init__()
        self.env = env
        self.config = env.config
        self.queue = multiprocessing.Queue()
        try:
            self.timer_scalar = utils.train.Timer(env.config.getfloat('summary', 'scalar'))
        except configparser.NoOptionError:
            self.timer_scalar = lambda: False
        try:
            self.timer_image = utils.train.Timer(env.config.getfloat('summary', 'image'))
        except configparser.NoOptionError:
            self.timer_image = lambda: False
        try:
            self.timer_histogram = utils.train.Timer(env.config.getfloat('summary', 'histogram'))
        except configparser.NoOptionError:
            self.timer_histogram = lambda: False
        with open(os.path.expanduser(os.path.expandvars(env.config.get('summary_histogram', 'parameters'))), 'r') as f:
            self.histogram_parameters = utils.RegexList([line.rstrip() for line in f])
        self.draw_bbox = utils.visualize.DrawBBox(env.category)
        self.draw_feature = utils.visualize.DrawFeature()

    def __call__(self, name, **kwargs):
        if getattr(self, 'timer_' + name)():
            kwargs = getattr(self, 'copy_' + name)(**kwargs)
            self.queue.put((name, kwargs))

    def stop(self):
        self.queue.put((None, {}))

    def run(self):
        self.writer = SummaryWriter(os.path.join(self.env.model_dir, self.env.args.run))
        while True:
            name, kwargs = self.queue.get()
            if name is None:
                break
            func = getattr(self, 'summary_' + name)
            try:
                func(**kwargs)
            except:
                traceback.print_exc()

    def copy_scalar(self, **kwargs):
        step, loss_total, loss, loss_hparam = (kwargs[key] for key in 'step, loss_total, loss, loss_hparam'.split(', '))
        loss_total = loss_total.data.clone().cpu().numpy()
        loss = {key: l.data.clone().cpu().numpy() for key, l in loss.items()}
        loss_hparam = {key: l.data.clone().cpu().numpy() for key, l in loss_hparam.items()}
        return dict(
            step=step,
            loss_total=loss_total,
            loss=loss, loss_hparam=loss_hparam,
        )

    def summary_scalar(self, **kwargs):
        step, loss_total, loss, loss_hparam = (kwargs[key] for key in 'step, loss_total, loss, loss_hparam'.split(', '))
        for key, l in loss.items():
            self.writer.add_scalar('loss/' + key, l[0], step)
        if self.config.getboolean('summary_scalar', 'loss_hparam'):
            self.writer.add_scalars('loss_hparam', {key: l[0] for key, l in loss_hparam.items()}, step)
        self.writer.add_scalar('loss_total', loss_total[0], step)

    def copy_image(self, **kwargs):
        step, height, width, rows, cols, data, pred, debug = (kwargs[key] for key in 'step, height, width, rows, cols, data, pred, debug'.split(', '))
        data = {key: data[key].clone().cpu().numpy() for key in 'image, yx_min, yx_max, cls'.split(', ')}
        pred = {key: pred[key].data.clone().cpu().numpy() for key in 'yx_min, yx_max, iou, logits'.split(', ') if key in pred}
        matching = (debug['positive'].float() - debug['negative'].float() + 1) / 2
        matching = matching.data.clone().cpu().numpy()
        return dict(
            step=step, height=height, width=width, rows=rows, cols=cols,
            data=data, pred=pred,
            matching=matching,
        )

    def summary_image(self, **kwargs):
        step, height, width, rows, cols, data, pred, matching = (kwargs[key] for key in 'step, height, width, rows, cols, data, pred, matching'.split(', '))
        image = data['image']
        limit = min(self.config.getint('summary_image', 'limit'), image.shape[0])
        image = image[:limit, :, :, :]
        yx_min, yx_max, iou = (pred[key] for key in 'yx_min, yx_max, iou'.split(', '))
        scale = [height / rows, width / cols]
        yx_min, yx_max = (a * scale for a in (yx_min, yx_max))
        if 'logits' in pred:
            cls = np.argmax(F.softmax(torch.autograd.Variable(torch.from_numpy(pred['logits'])), -1).data.cpu().numpy(), -1)
        else:
            cls = np.zeros(iou.shape, np.int)
        if self.config.getboolean('summary_image', 'bbox'):
            # data
            canvas = np.copy(image)
            canvas = pybenchmark.profile('bbox/data')(self.draw_bbox_data)(canvas, *(data[key] for key in 'yx_min, yx_max, cls'.split(', ')))
            self.writer.add_image('bbox/data', torchvision.utils.make_grid(torch.from_numpy(np.stack(canvas)).permute(0, 3, 1, 2).float(), normalize=True, scale_each=True), step)
            # pred
            canvas = np.copy(image)
            canvas = pybenchmark.profile('bbox/pred')(self.draw_bbox_pred)(canvas, yx_min, yx_max, cls, iou, nms=True)
            self.writer.add_image('bbox/pred', torchvision.utils.make_grid(torch.from_numpy(np.stack(canvas)).permute(0, 3, 1, 2).float(), normalize=True, scale_each=True), step)
        if self.config.getboolean('summary_image', 'iou'):
            # bbox
            canvas = np.copy(image)
            canvas_data = self.draw_bbox_data(canvas, *(data[key] for key in 'yx_min, yx_max, cls'.split(', ')), colors=['g'])
            # data
            for i, canvas in enumerate(pybenchmark.profile('iou/data')(self.draw_bbox_iou)(list(map(np.copy, canvas_data)), yx_min, yx_max, cls, matching, rows, cols, colors=['w'])):
                canvas = np.stack(canvas)
                canvas = torch.from_numpy(canvas).permute(0, 3, 1, 2)
                canvas = torchvision.utils.make_grid(canvas.float(), normalize=True, scale_each=True)
                self.writer.add_image('iou/data%d' % i, canvas, step)
            # pred
            for i, canvas in enumerate(pybenchmark.profile('iou/pred')(self.draw_bbox_iou)(list(map(np.copy, canvas_data)), yx_min, yx_max, cls, iou, rows, cols, colors=['w'])):
                canvas = np.stack(canvas)
                canvas = torch.from_numpy(canvas).permute(0, 3, 1, 2)
                canvas = torchvision.utils.make_grid(canvas.float(), normalize=True, scale_each=True)
                self.writer.add_image('iou/pred%d' % i, canvas, step)

    def draw_bbox_data(self, canvas, yx_min, yx_max, cls, colors=None):
        batch_size = len(canvas)
        if len(cls.shape) == len(yx_min.shape):
            cls = np.argmax(cls, -1)
        yx_min, yx_max, cls = ([a[b] for b in range(batch_size)] for a in (yx_min, yx_max, cls))
        return [self.draw_bbox(canvas, yx_min.astype(np.int), yx_max.astype(np.int), cls, colors=colors) for canvas, yx_min, yx_max, cls in zip(canvas, yx_min, yx_max, cls)]

    def draw_bbox_pred(self, canvas, yx_min, yx_max, cls, iou, colors=None, nms=False):
        batch_size = len(canvas)
        mask = iou > self.config.getfloat('detect', 'threshold')
        yx_min, yx_max = (np.reshape(a, [a.shape[0], -1, 2]) for a in (yx_min, yx_max))
        cls, iou, mask = (np.reshape(a, [a.shape[0], -1]) for a in (cls, iou, mask))
        yx_min, yx_max, cls, iou, mask = ([a[b] for b in range(batch_size)] for a in (yx_min, yx_max, cls, iou, mask))
        yx_min, yx_max, cls, iou = ([a[m] for a, m in zip(l, mask)] for l in (yx_min, yx_max, cls, iou))
        if nms:
            overlap = self.config.getfloat('detect', 'overlap')
            keep = [pybenchmark.profile('nms')(utils.postprocess.nms)(torch.Tensor(iou), torch.Tensor(yx_min), torch.Tensor(yx_max), overlap) if iou.shape[0] > 0 else [] for yx_min, yx_max, iou in zip(yx_min, yx_max, iou)]
            keep = [np.array(k, np.int) for k in keep]
            yx_min, yx_max, cls = ([a[k] for a, k in zip(l, keep)] for l in (yx_min, yx_max, cls))
        return [self.draw_bbox(canvas, yx_min.astype(np.int), yx_max.astype(np.int), cls, colors=colors) for canvas, yx_min, yx_max, cls in zip(canvas, yx_min, yx_max, cls)]

    def draw_bbox_iou(self, canvas_share, yx_min, yx_max, cls, iou, rows, cols, colors=None):
        batch_size = len(canvas_share)
        yx_min, yx_max = ([np.squeeze(a, -2) for a in np.split(a, a.shape[-2], -2)] for a in (yx_min, yx_max))
        cls, iou = ([np.squeeze(a, -1) for a in np.split(a, a.shape[-1], -1)] for a in (cls, iou))
        results = []
        for i, (yx_min, yx_max, cls, iou) in enumerate(zip(yx_min, yx_max, cls, iou)):
            mask = iou > self.config.getfloat('detect', 'threshold')
            yx_min, yx_max = (np.reshape(a, [a.shape[0], -1, 2]) for a in (yx_min, yx_max))
            cls, iou, mask = (np.reshape(a, [a.shape[0], -1]) for a in (cls, iou, mask))
            yx_min, yx_max, cls, iou, mask = ([a[b] for b in range(batch_size)] for a in (yx_min, yx_max, cls, iou, mask))
            yx_min, yx_max, cls = ([a[m] for a, m in zip(l, mask)] for l in (yx_min, yx_max, cls))
            canvas = [self.draw_bbox(canvas, yx_min.astype(np.int), yx_max.astype(np.int), cls, colors=colors) for canvas, yx_min, yx_max, cls in zip(np.copy(canvas_share), yx_min, yx_max, cls)]
            iou = [np.reshape(a, [rows, cols]) for a in iou]
            canvas = [self.draw_feature(_canvas, iou) for _canvas, iou in zip(canvas, iou)]
            results.append(canvas)
        return results

    def copy_histogram(self, **kwargs):
        return {key: kwargs[key].data.clone().cpu().numpy() if torch.is_tensor(kwargs[key]) else kwargs[key] for key in 'step, dnn'.split(', ')}

    def summary_histogram(self, **kwargs):
        step, dnn = (kwargs[key] for key in 'step, dnn'.split(', '))
        for name, param in dnn.named_parameters():
            if self.histogram_parameters(name):
                self.writer.add_histogram(name, param, step)


class Train(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.model_dir = utils.get_model_dir(config)
        self.cache_dir = utils.get_cache_dir(config)
        self.category = utils.get_category(config, self.cache_dir)
        self.anchors = torch.from_numpy(utils.get_anchors(config)).contiguous()
        logging.info('use cache directory ' + self.cache_dir)
        logging.info('tensorboard --logdir ' + self.model_dir)
        if args.delete:
            logging.warning('delete model directory: ' + self.model_dir)
            shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_dir + '.ini', 'w') as f:
            config.write(f)
        self.saver = utils.train.Saver(self.model_dir, config.getint('save', 'keep'))
        self.timer_save = utils.train.Timer(config.getfloat('save', 'secs'), False)
        try:
            self.timer_eval = utils.train.Timer(eval(config.get('eval', 'secs')), config.getboolean('eval', 'first'))
        except configparser.NoOptionError:
            self.timer_eval = lambda: False
        self.summary_worker = SummaryWorker(self)
        self.summary_worker.start()

    def stop(self):
        self.summary_worker.stop()
        self.summary_worker.join()

    def get_loader(self):
        paths = [os.path.join(self.cache_dir, phase + '.pkl') for phase in self.config.get('train', 'phase').split()]
        dataset = utils.data.Dataset(
            utils.data.load_pickles(paths),
            transform=transform.augmentation.get_transform(self.config, self.config.get('transform', 'augmentation').split()),
            one_hot=None if self.config.getboolean('train', 'cross_entropy') else len(self.category),
            shuffle=self.config.getboolean('data', 'shuffle'),
            dir=os.path.join(self.model_dir, 'exception'),
        )
        logging.info('num_examples=%d' % len(dataset))
        try:
            workers = self.config.getint('data', 'workers')
            if torch.cuda.is_available():
                workers = workers * torch.cuda.device_count()
        except configparser.NoOptionError:
            workers = multiprocessing.cpu_count()
        collate_fn = utils.data.Collate(
            transform.parse_transform(self.config, self.config.get('transform', 'resize_train')),
            utils.train.load_sizes(self.config),
            maintain=self.config.getint('data', 'maintain'),
            transform_image=transform.get_transform(self.config, self.config.get('transform', 'image_train').split()),
            transform_tensor=transform.get_transform(self.config, self.config.get('transform', 'tensor').split()),
            dir=os.path.join(self.model_dir, 'exception'),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size * torch.cuda.device_count() if torch.cuda.is_available() else self.args.batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    def load(self):
        try:
            path, step, epoch = utils.train.load_model(self.model_dir)
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            config_channels = model.ConfigChannels(self.config, state_dict)
        except ValueError:
            step, epoch = 0, 0
            config_channels = model.ConfigChannels(self.config)
        dnn = utils.parse_attr(self.config.get('model', 'dnn'))(config_channels, self.anchors, len(self.category))
        if config_channels.state_dict is not None:
            dnn.load_state_dict(config_channels.state_dict)
        return step, epoch, dnn

    def finetune(self, model, path):
        if os.path.isdir(path):
            path, _step, _epoch = utils.train.load_model(path)
        _state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = model.state_dict()
        ignore = utils.RegexList(self.args.ignore)
        for key, value in state_dict.items():
            try:
                if not ignore(key):
                    state_dict[key] = _state_dict[key]
            except KeyError:
                logging.warning('%s not in finetune file %s' % (key, path))
        model.load_state_dict(state_dict)

    def step(self, inference, optimizer, data):
        for key in data:
            t = data[key]
            if torch.is_tensor(t):
                data[key] = utils.ensure_device(t)
        tensor = torch.autograd.Variable(data['tensor'])
        pred = pybenchmark.profile('inference')(model._inference)(inference, tensor)
        height, width = data['image'].size()[1:3]
        rows, cols = pred['feature'].size()[-2:]
        loss, debug = pybenchmark.profile('loss')(model.loss)(self.anchors, norm_data(data, height, width, rows, cols), pred, self.config.getfloat('model', 'threshold'))
        loss_hparam = {key: loss[key] * self.config.getfloat('hparam', key) for key in loss}
        loss_total = sum(loss_hparam.values())
        optimizer.zero_grad()
        loss_total.backward()
        try:
            clip = self.config.getfloat('train', 'clip')
            nn.utils.clip_grad_norm(inference.parameters(), clip)
        except configparser.NoOptionError:
            pass
        optimizer.step()
        return dict(
            height=height, width=width, rows=rows, cols=cols,
            data=data, pred=pred, debug=debug,
            loss_total=loss_total, loss=loss, loss_hparam=loss_hparam,
        )

    def __call__(self):
        with filelock.FileLock(os.path.join(self.model_dir, 'lock'), 0):
            try:
                step, epoch, dnn = self.load()
                inference = model.Inference(self.config, dnn, self.anchors)
                logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in inference.state_dict().values())))
                if self.args.finetune:
                    path = os.path.expanduser(os.path.expandvars(self.args.finetune))
                    logging.info('finetune from ' + path)
                    self.finetune(dnn, path)
                inference = ensure_model(inference)
                inference.train()
                optimizer = eval(self.config.get('train', 'optimizer'))(filter(lambda p: p.requires_grad, inference.parameters()), self.args.learning_rate)
                try:
                    scheduler = eval(self.config.get('train', 'scheduler'))(optimizer)
                except configparser.NoOptionError:
                    scheduler = None
                loader = self.get_loader()
                logging.info('num_workers=%d' % loader.num_workers)
                for epoch in range(0 if epoch is None else epoch, self.args.epoch):
                    if scheduler is not None:
                        scheduler.step(epoch)
                        logging.info('epoch=%d, lr=%s' % (epoch, str(scheduler.get_lr())))
                    for data in loader if self.args.quiet else tqdm.tqdm(loader, desc='epoch=%d/%d' % (epoch, self.args.epoch)):
                        kwargs = self.step(inference, optimizer, data)
                        step += 1
                        kwargs = {**kwargs, **dict(
                            dnn=dnn, inference=inference, optimizer=optimizer,
                            step=step, epoch=epoch,
                        )}
                        self.summary_worker('scalar', **kwargs)
                        self.summary_worker('image', **kwargs)
                        self.summary_worker('histogram', **kwargs)
                        if self.timer_save():
                            self.save(**kwargs)
                        if self.timer_eval():
                            self.eval(**kwargs)
                self.save(**kwargs)
                logging.info('finished')
            except KeyboardInterrupt:
                logging.warning('interrupted')
                self.save(**kwargs)
            except:
                traceback.print_exc()
                try:
                    with open(os.path.join(self.model_dir, 'data.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                except UnboundLocalError:
                    pass
                raise
            finally:
                self.stop()

    def check_nan(self, **kwargs):
        step, loss_total, loss, data = (kwargs[key] for key in 'step, loss_total, loss, data'.split(', '))
        if np.isnan(loss_total.data.cpu()[0]):
            dump_dir = os.path.join(self.model_dir, str(step))
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(collections.OrderedDict([(key, var.cpu()) for key, var in kwargs['dnn'].state_dict().items()]), os.path.join(dump_dir, 'model.pth'))
            torch.save(data, os.path.join(dump_dir, 'data.pth'))
            for key, l in loss.items():
                logging.warning('%s=%f' % (key, l.data.cpu()[0]))
            raise OverflowError('NaN loss detected, dump runtime information into ' + dump_dir)

    def save(self, **kwargs):
        step, epoch = (kwargs[key] for key in 'step, epoch'.split(', '))
        self.check_nan(**kwargs)
        self.saver(collections.OrderedDict([(key, var.cpu()) for key, var in kwargs['dnn'].state_dict().items()]), step, epoch)

    def eval(self, **kwargs):
        step, inference = (kwargs[key] for key in 'step, inference'.split(', '))
        logging.info('evaluating')
        if torch.cuda.is_available():
            inference.cpu()
        try:
            e = _eval.Eval(self.args, self.config)
            cls_ap = e()
            self.backup_best(cls_ap, e.path)
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            inference.cuda()

    def backup_best(self, cls_ap, path):
        try:
            with open(self.model_dir + '.pkl', 'rb') as f:
                best = np.mean(list(pickle.load(f).values()))
        except:
            best = np.finfo(np.float32).min
        metric = np.mean(list(cls_ap.values()))
        if metric > best:
            with open(self.model_dir + '.pkl', 'wb') as f:
                pickle.dump(cls_ap, f)
            shutil.copy(path, self.model_dir + '.pth')
            logging.info('best model (%f) saved into %s.*' % (metric, self.model_dir))
        else:
            logging.info('best metric %f >= %f' % (best, metric))


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    if args.run is None:
        buffer = io.StringIO()
        config.write(buffer)
        args.run = hashlib.md5(buffer.getvalue().encode()).hexdigest()
    logging.info('cd ' + os.getcwd() + ' && ' + subprocess.list2cmdline([sys.executable] + sys.argv))
    train = Train(args, config)
    train()
    logging.info(pybenchmark.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('-f', '--finetune')
    parser.add_argument('-i', '--ignore', nargs='+', default=[], help='regex to ignore weights while fintuning')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-e', '--epoch', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('-d', '--delete', action='store_true', help='delete model')
    parser.add_argument('-q', '--quiet', action='store_true', help='quiet mode')
    parser.add_argument('-r', '--run', help='the run name in TensorBoard')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
