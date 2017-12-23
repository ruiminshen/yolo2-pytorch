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
import random


def main():
    args = make_args()
    root = os.path.expanduser(os.path.expandvars(args.root))
    realpaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[-1].lower() in args.exts and filename[0] != '.':
                path = os.path.join(dirpath, filename)
                realpath = os.path.relpath(path, root)
                realpaths.append(realpath)
    random.shuffle(realpaths)
    total = args.train + args.val + args.test
    nval = int(len(realpaths) * args.val / total)
    ntest = nval + int(len(realpaths) * args.test / total)
    val = realpaths[:nval]
    test = realpaths[nval:ntest]
    train = realpaths[ntest:]
    print('train=%d, val=%d, test=%d' % (len(train), len(val), len(test)))
    with open(os.path.join(root, 'train' + args.ext), 'w') as f:
        for path in train:
            f.write(path + '\n')
    with open(os.path.join(root, 'val' + args.ext), 'w') as f:
        for path in val:
            f.write(path + '\n')
    with open(os.path.join(root, 'test' + args.ext), 'w') as f:
        for path in test:
            f.write(path + '\n')


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('-e', '--exts', nargs='+', default=['.jpg', '.png'])
    parser.add_argument('--train', type=float, default=7)
    parser.add_argument('--val', type=float, default=2)
    parser.add_argument('--test', type=float, default=1)
    parser.add_argument('--ext', default='.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main()
