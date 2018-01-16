import os
import sys
import argparse
import hashlib
import requests
import mimetypes
import threading
import traceback

import numpy as np
import tqdm


def _task(root_image, root_url, url):
    name = hashlib.md5(url.encode()).hexdigest()
    response = requests.get(url)
    content_type = response.headers['content-type']
    ext = mimetypes.guess_extension(content_type, False)
    path = os.path.join(root_image, name) + ext
    if not os.path.exists(path):
        try:
            with open(path, 'wb') as f:
                for data in response.iter_content():
                    f.write(data)
        except:
            os.remove(path)
            raise
    with open(os.path.join(root_url, name), 'w') as f:
        f.write(url)


def task(root_image, root_url, urls, pbar):
    for url in urls:
        try:
            _task(root_image, root_url, url)
        except:
            traceback.print_exc()
        pbar.update()


def main():
    args = make_args()
    root = os.path.expandvars(os.path.expanduser(args.root))
    root_image = os.path.join(root, args.dir_image)
    root_url = os.path.join(root, args.dir_url)
    os.makedirs(root_image, exist_ok=True)
    os.makedirs(root_url, exist_ok=True)
    workers = []
    urls = sys.stdin.readlines()
    with tqdm.tqdm(total=len(urls)) as pbar:
        for urls in np.array_split(urls, args.workers):
            w = threading.Thread(target=task, args=(root_image, root_url, urls, pbar))
            w.start()
            workers.append(w)
        for w in workers:
            w.join()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('-w', '--workers', type=int, default=6)
    parser.add_argument('--dir_image', default='image')
    parser.add_argument('--dir_url', default='url')
    return parser.parse_args()


if __name__ == '__main__':
    main()
