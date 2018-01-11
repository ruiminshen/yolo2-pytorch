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
import sys
import argparse
import shutil
import tqdm

import cv2


def main():
    args = make_args()
    root = os.path.expanduser(os.path.expandvars(args.root))
    for dirpath, _, filenames in os.walk(root):
        for filename in tqdm.tqdm(filenames, desc=dirpath):
            if os.path.splitext(filename)[-1].lower() in args.exts and filename[0] != '.':
                path = os.path.join(dirpath, filename)
                image = cv2.imread(path)
                if image is None:
                    sys.stderr.write('disable bad image %s\n' % path)
                    shutil.move(path, os.path.join(os.path.dirname(path), '.' + os.path.basename(path)))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-e', '--exts', nargs='+', default=['.jpe', '.jpg', '.jpeg', '.png'])
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()


if __name__ == '__main__':
    main()
