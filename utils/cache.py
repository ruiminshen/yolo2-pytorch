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

import numpy as np


def verify_coords(yx_min, yx_max, size):
    assert np.all(yx_min <= yx_max), 'yx_min <= yx_max'
    assert np.all(0 <= yx_min), '0 <= yx_min'
    assert np.all(0 <= yx_max), '0 <= yx_max'
    assert np.all(yx_min < size), 'yx_min < size'
    assert np.all(yx_max < size), 'yx_max < size'


def fix_coords(yx_min, yx_max, size):
    assert np.all(yx_min <= yx_max)
    assert yx_min.dtype == yx_max.dtype
    coord_min = np.zeros([2], dtype=yx_min.dtype)
    coord_max = np.array(size, dtype=yx_min.dtype) - 1
    yx_min = np.minimum(np.maximum(yx_min, coord_min), coord_max)
    yx_max = np.minimum(np.maximum(yx_max, coord_min), coord_max)
    return yx_min, yx_max
