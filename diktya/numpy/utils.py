# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from math import ceil
from math import sqrt
from scipy.misc import toimage, bytescale


def scipy_gaussian_filter_2d(x, sigma=2/3.):
    from scipy.ndimage.filters import gaussian_filter1d
    return gaussian_filter1d(gaussian_filter1d(x, sigma, axis=-1), sigma, axis=-2)


def tile(tiles, columns_must_be_multiple_of=1, border=0, border_value=0):
    def calc_columns_rows(n):
        num_columns = int(ceil(sqrt(n)))
        if num_columns % columns_must_be_multiple_of != 0:
            num_columns += columns_must_be_multiple_of - \
                num_columns % columns_must_be_multiple_of
        num_rows = int(ceil(n / float(num_columns)))
        return num_columns, num_rows

    cols, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].shape

    if len(tile_size) == 2:
        tile_height, tile_width = tile_size
        tile_height += border
        tile_width += border
        combined_size = (1, tile_height * rows, tile_width * cols)
    elif len(tile_size) == 3:
        tile_height, tile_width = tile_size[1:]
        tile_height += border
        tile_width += border
        combined_size = (tile_size[0], tile_height * rows, tile_width * cols)
    else:
        raise ValueError("Only 2 or 3 dim input size are supported, got: {}"
                         .format(tile_size))
    im = border_value * np.ones(combined_size)
    for r in range(rows):
        for c in range(cols):
            tile_idx = r*cols + c
            if tile_idx < len(tiles):
                ir = r*tile_height
                ic = c*tile_width
                im[:, ir:ir+tile_height-border, ic:ic+tile_width-border] = tiles[tile_idx]
    return im


def zip_tile(*arrs, columns_must_be_multiple_of=None):
    assert len(arrs) >= 1
    length = len(arrs[0])
    for a in arrs:
        assert len(a) == length, "all input arrays must have the same size"
    tiles = []
    for i in range(length):
        for a in arrs:
            tiles.append(a[i])

    if columns_must_be_multiple_of is None:
        columns_must_be_multiple_of = len(arrs)
    tiled = tile(tiles, columns_must_be_multiple_of)
    assert len(tiled) == 1, "only grayscale image are supported"
    return tiled[0]


def image_save(fname, img_array, low=-1, high=1):
    """
    Saves the image ``img_array`` to ``fname``.

    Args:
        fname: Save image under this filename
        img_array: array of shape ``(height, width)``, ``(1, height, width)``,
            or ``(3, height, width)``.
        low: Lowest pixel value
        high: Highest pixel value.

    The image range must be between ``low`` and ``high``.
    """
    if img_array.dtype != np.uint8:
        if img_array.min() < low:
            raise Exception("Got an image with min {}, but low is set to {}."
                            .format(img_array.min(), low))

        if img_array.max() > high:
            raise Exception("Got an image with max {}, but high is set to {}."
                            .format(img_array.min(), low))

    ndim = len(img_array.shape)
    nb_channels = len(img_array)
    if ndim == 3 and nb_channels == 1:
        img = img_array[0]
    elif ndim == 3 and nb_channels == 3:
        img = np.moveaxis(img_array, 0, -1)
    elif ndim == 2:
        img = img_array
    else:
        raise Exception("Did not understand image shape: {}".format(img_array.shape))

    if img_array.dtype == np.uint8:
        img_bytes = img_array
    else:
        img_0_to_1 = (img - low) / (high - low)
        img_bytes = bytescale(img_0_to_1 * 255, cmin=0, cmax=255)
    toimage(img_bytes).save(fname)
