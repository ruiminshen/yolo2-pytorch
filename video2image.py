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

import pymediainfo
import tqdm
from contextlib import closing
import videosequence


def get_step(frames, video_track, **kwargs):
    if 'frames' in kwargs:
        step = len(frames) // kwargs['frames']
    elif 'frames_per_sec' in kwargs > 0:
        frame_rate = float(video_track.frame_rate)
        step = int(frame_rate / kwargs['frames_per_sec'])
    assert step > 0
    return step


def convert(video_file, image_prefix, **kwargs):
    media_info = pymediainfo.MediaInfo.parse(video_file)
    video_tracks = [track for track in media_info.tracks if track.track_type == 'Video']
    if len(video_tracks) < 1:
        raise videosequence.VideoError()
    video_track = video_tracks[0]
    _rotation = float(video_track.rotation)
    rotation = int(_rotation)
    assert rotation - _rotation == 0
    with closing(videosequence.VideoSequence(video_file)) as frames:
        step = get_step(frames, video_track, **kwargs)
        _frames = frames[::step]
        for idx, frame in enumerate(tqdm.tqdm(_frames)):
            frame = frame.rotate(-rotation, expand=True)
            frame.save('%s_%04d.jpg' % (image_prefix, idx))


def main():
    args = make_args()
    src = os.path.expanduser(os.path.expandvars(args.src))
    dst = os.path.expanduser(os.path.expandvars(args.dst))
    os.makedirs(dst, exist_ok=True)
    kwargs = {}
    if args.frames > 0:
        kwargs['frames'] = args.frames
    elif args.frames_per_sec > 0:
        kwargs['frames_per_sec'] = args.frames_per_sec
    exts = set()
    for dirpath, _, filenames in os.walk(src):
        for filename in filenames:
            ext = os.path.splitext(filename)[-1].lower()
            if ext in args.ext:
                path = os.path.join(dirpath, filename)
                print(path)
                name = os.path.relpath(path, src).replace(os.path.sep, args.replace)
                _path = os.path.join(dst, name)
                try:
                    convert(path, _path, **kwargs)
                except videosequence.VideoError as e:
                    sys.stderr.write(str(e) + '\n')
            else:
                exts.add(ext)
    print(exts)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('-e', '--ext', nargs='+', default=['.mp4', '.mov', '.m4v'])
    parser.add_argument('-r', '--replace', default='_', help='replace the path separator into the given character')
    parser.add_argument('-f', '--frames', default=0, type=int, help='total output frames in a video')
    parser.add_argument('--frames_per_sec', default=0, type=int, help='output frames in a second')
    return parser.parse_args()


if __name__ == '__main__':
    main()
