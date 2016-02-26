# Copyright 2015 Leon Sixt
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
import os
from math import ceil
from math import sqrt

import keras
import skimage
from keras.callbacks import Callback
from keras.layers.convolutional import Convolution2D
import numpy as np
import theano
import theano.tensor as T
from keras.layers.core import Layer
from keras.backend.common import cast_to_floatx, floatx
from scipy.misc import imsave
import skimage.filters


class LossPrinter(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        print("#{}-{} ".format(self.epoch, batch), end='')
        for k in self.params['metrics']:
            if k in logs:
                print(" {}: {:.4f}".format(k, float(logs[k])), end='')
        print('')


class Sample(keras.callbacks.Callback):
    def __init__(self, outdir, every_nth_epoch=10):
        super().__init__()
        self.outdir = outdir
        self.every_nth_epoch = every_nth_epoch

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.every_nth_epoch != 0:
            return
        sample = self.model.generate()
        out_dir = os.path.abspath(
            os.path.join(self.outdir, "epoche_{}/".format(epoch)))
        print("Writing {} samples to: {}".format(len(sample), out_dir))
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(sample)):
            outpath = os.path.join(out_dir, str(i) + ".png")

            imsave(outpath,
                   (sample[i]*255).reshape(3, 16, 16).astype(np.uint8))


_gaussian_blur_kernel = np.asarray([[[
    [1., 4., 6., 4., 1.],
    [4., 16., 24., 16., 4.],
    [6., 24., 36., 24., 6.],
    [4., 16., 24., 16., 4.],
    [1., 4., 6., 4., 1.]
]]], dtype=floatx())


def upsample(input, sigma=4/6, nb_channels=1):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]

    upsample_layer_weight = gaussian_kernel(sigma)
    kernel_size = upsample_layer_weight.shape[-1]
    upsample_layer = Convolution2D(nb_channels, kernel_size, kernel_size,
                                   border_mode='valid',
                                   input_shape=(nb_channels, None, None))
    upsample_layer_weight = upsample_layer_weight.repeat(nb_channels, axis=1)
    upsample_layer.W = theano.shared(upsample_layer_weight)
    shp = input.shape
    upsampled = T.zeros((shp[0], shp[1], 2*shp[2], 2*shp[3]))
    upsampled = T.set_subtensor(upsampled[:, :, ::2, ::2], input)
    upsampled = _add_virtual_border(upsampled)
    upsample_layer.input = upsampled
    return upsample_layer.get_output(train=False)


def gaussian_kernel(sigma, size=None, nb_channels=1):
    if size is None:
        size = 2*ceil(2*sigma)+1
    a = np.zeros((size, size))
    center = size // 2
    a[center, center] = 1
    gaussian_kernel = skimage.filters.gaussian_filter(a, sigma)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis]
    gaussian_kernel = gaussian_kernel.repeat(nb_channels, axis=1)
    return gaussian_kernel.astype(np.float32)


def resize_nearest_neighbour(images, scale):
    assert type(scale) == int
    assert scale >= 1
    return images[:, :, ::scale, ::scale]


def resize_interpolate(input, scale):
    # from here https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py
    num_batch, num_channels, height, width = input.shape

    out_height = T.cast(height / scale, 'int64')
    out_width = T.cast(width / scale, 'int64')
    theta = theano.shared(np.array([[[1, 0, 0],
                                     [0, 1, 0]]]))
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


def smooth(input, sigma=2/4, add_border=True, nb_channels=1):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]
    kernel = gaussian_kernel(sigma, nb_channels=nb_channels)
    size = kernel.shape[-1]
    if add_border:
        input = _add_virtual_border(input, filter_size=size)
    smooth_layer = Convolution2D(nb_channels, size, size, border_mode='valid',
                                 input_shape=(1, None, None))
    smooth_layer.W = theano.shared(cast_to_floatx(kernel))
    smooth_layer.input = input
    return smooth_layer.get_output(train=False)


class BorderReflect(Layer):
    def __init__(self, filter_size=5, **kwargs):
        super().__init__(**kwargs)
        self.filter_size = filter_size

    def get_output(self, train=False):
        return _add_virtual_border(self.get_input(train), self.filter_size)

    @property
    def output_shape(self):
        shp = self.input_shape
        f = self.filter_size
        return shp[:2] + (shp[2] + f - 1, shp[3] + f - 1)


def _add_virtual_border(input, filter_size=5):
    """Reflects the border like OpenCV BORDER_REFLECT_101. See here
    http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html"""

    shp = input.shape
    assert filter_size % 2 == 1, "only works with odd filters"
    half = (filter_size - 1) // 2
    wb = T.zeros((shp[0], shp[1], shp[2]+2*half, shp[3]+2*half))
    wb = T.set_subtensor(wb[:, :, half:shp[2]+half, half:shp[3]+half], input)

    top = input[:, :, 1:half+1, :]
    wb = T.set_subtensor(wb[:, :, :half, half:shp[3]+half], top[:, :, ::-1, :])

    bottom = input[:, :, -half-1:-1, :]
    wb = T.set_subtensor(wb[:, :, -half:, half:shp[3]+half], bottom[:, :, ::-1, :])

    left = input[:, :, :, 1:half+1]
    wb = T.set_subtensor(wb[:, :, half:shp[2]+half, :half], left[:, :, :, ::-1])

    right = input[:, :, :, -half-1:-1]
    wb = T.set_subtensor(wb[:, :, half:shp[2]+half, -half:], right[:, :, :, ::-1])

    left_top = input[:, :, 1:half+1, 1:half+1]
    wb = T.set_subtensor(wb[:, :, :half, :half], left_top[:, :, ::-1, ::-1])
    left_bottom = input[:, :, -half-1:-1, 1:half+1]
    wb = T.set_subtensor(wb[:, :, -half:, :half], left_bottom[:, :, ::-1, ::-1])
    right_bottom = input[:, :, 1:half+1, -half-1:-1]
    wb = T.set_subtensor(wb[:, :, :half, -half:], right_bottom[:, :, ::-1, ::-1])
    right_top = input[:, :, -half-1:-1, -half-1:-1]
    wb = T.set_subtensor(wb[:, :, -half:, -half:], right_top[:, :, ::-1, ::-1])
    return wb


def sobel(img):
    conv_x = Convolution2D(1, 3, 3, border_mode='valid',
                           input_shape=(1, None, None))
    conv_y = Convolution2D(1, 3, 3, border_mode='valid',
                           input_shape=(1, None, None))
    filter = cast_to_floatx(np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]))
    conv_x.W = theano.shared(filter[np.newaxis, np.newaxis])
    conv_y.W = theano.shared(np.transpose(filter)[np.newaxis, np.newaxis])
    conv_x.input = img
    conv_y.input = img
    return conv_x.get_output(train=False), conv_y.get_output(train=False)


def downsample(input, nb_channels=1):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]

    downsample_layer = Convolution2D(nb_channels, 5, 5, subsample=(2, 2),
                                     border_mode='valid',
                                     input_shape=(1, None, None))
    downsample_layer_weight = _gaussian_blur_kernel/256.
    downsample_layer_weight = downsample_layer_weight \
        .repeat(nb_channels, axis=1)
    downsample_layer.W = theano.shared(downsample_layer_weight)
    input = _add_virtual_border(input)
    downsample_layer.input = input
    return downsample_layer.get_output(train=False)


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
