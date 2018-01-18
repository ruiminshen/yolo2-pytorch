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
import threading

import numpy as np
import tqdm
import wget


def _task(url, root, ext):
    path = wget.download(url, bar=None)
    with open(path + ext, 'w') as f:
        f.write(url)


def task(urls, root, ext, pbar, lock, f):
    for url in urls:
        url = url.rstrip()
        try:
            _task(url, root, ext)
        except:
            with lock:
                f.write(url + '\n')
        pbar.update()


def main():
    args = make_args()
    root = os.path.expandvars(os.path.expanduser(args.root))
    os.makedirs(root, exist_ok=True)
    os.chdir(root)
    workers = []
    urls = list(set(sys.stdin.readlines()))
    lock = threading.Lock()
    with tqdm.tqdm(total=len(urls)) as pbar, open(root + args.ext, 'w') as f:
        for urls in np.array_split(urls, args.workers):
            w = threading.Thread(target=task, args=(urls, root, args.ext, pbar, lock, f))
            w.start()
            workers.append(w)
        for w in workers:
            w.join()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('-w', '--workers', type=int, default=6)
    parser.add_argument('-e', '--ext', default='.url')
    return parser.parse_args()


if __name__ == '__main__':
    main()
