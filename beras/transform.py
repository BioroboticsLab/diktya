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

from math import ceil
from math import sqrt

import numpy as np
import theano
import theano.tensor as T

from beras.filters import gaussian_filter_2d


def upsample(input, sigma=2/3):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]
    resized = resize_interpolate(input, scale=2)
    return gaussian_filter_2d(resized, sigma)


def tile(tiles, columns_must_be_multiple_of=1):
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
        combined_size = (1, tile_height * rows, tile_width * cols)
    elif len(tile_size) == 3:
        tile_height, tile_width = tile_size[1:]
        combined_size = (tile_size[0], tile_height * rows, tile_width * cols)
    else:
        raise ValueError("Only 2 or 3 dim input size are supported, got: {}"
                         .format(tile_size))
    im = np.zeros(combined_size)
    for r in range(rows):
        for c in range(cols):
            tile_idx = r*cols + c
            if tile_idx < len(tiles):
                ir = r*tile_height
                ic = c*tile_width
                im[:, ir:ir+tile_height, ic:ic+tile_width] = tiles[tile_idx]
    return im


def zip_tile(*arrs):
    assert len(arrs) >= 1
    length = len(arrs[0])
    for a in arrs:
        assert len(a) == length, "all input arrays must have the same size"
    tiles = []
    for i in range(length):
        for a in arrs:
            tiles.append(a[i])

    tiled = tile(tiles, columns_must_be_multiple_of=len(arrs))
    assert len(tiled) == 1, "only grayscale image are supported"
    return tiled[0]


def resize_nearest_neighbour(images, scale):
    assert type(scale) == int
    assert scale >= 1
    return images[:, :, ::scale, ::scale]


def resize_interpolate(input, scale):
    # from https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py
    num_batch, num_channels, height, width = input.shape

    out_height = T.cast(height * scale, 'int64')
    out_width = T.cast(width * scale, 'int64')
    theta = theano.shared(np.array([[[1, 0, 0],
                                     [0, 1, 0]]],
                                   dtype=theano.config.floatX))
    theta = theta.repeat(num_batch, axis=0)
    grid = _meshgrid(out_height, out_width)
    # this could be removed
    grid_t = T.dot(theta, grid)
    x_s = grid_t[:, 0]
    y_s = grid_t[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _interpolate(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)

    assert str(output.dtype) == theano.config.floatX, str(output.dtype)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


def scipy_gaussian_filter_2d(x, sigma):
    from scipy.ndimage.filters import gaussian_filter1d
    return gaussian_filter1d(gaussian_filter1d(x, 2/3, axis=-1), 2/3, axis=-2)
