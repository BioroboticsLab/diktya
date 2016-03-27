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
import keras.backend as K

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


def upsample(input, sigma=2/3, nb_channels=1):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]
    resized = resize_interpolate(input, scale=0.5)
    return smooth(resized, sigma, nb_channels=nb_channels)


def np_gaussian_kernel(sigma, size=None, nb_channels=1):
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
    # from https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py
    num_batch, num_channels, height, width = input.shape

    out_height = T.cast(height / scale, 'int64')
    out_width = T.cast(width / scale, 'int64')
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


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def gaussian_kernel_default_radius(sigma, window_radius=None):
    if window_radius is None:
        return T.ceil(3*sigma)
    else:
        return window_radius


@static_vars(cache={})
def gaussian_kernel_1d(sigma, window_radius=None):
    if (sigma, window_radius) in gaussian_kernel_1d.cache:
        return gaussian_kernel_1d.cache[(sigma, window_radius)]

    radius = gaussian_kernel_default_radius(sigma, window_radius)
    index = T.arange(2*radius + 1) - radius
    kernel = T.exp(-0.5*index**2/sigma**2)
    kernel = kernel / kernel.sum()

    if type(sigma) in (float, int) and \
            type(window_radius) in (float, int, type(None)):
        gaussian_kernel_1d.cache[(sigma, window_radius)] = kernel
    return T.cast(kernel, 'float32')


def add_border(input, border, mode='repeat'):
    if mode == 'repeat':
        return add_border_repeat(input, border)
    elif mode == 'reflect':
        return add_border_reflect(input, border)
    else:
        raise ValueError("Invalid mode: {}".format(mode))


def add_border_repeat(input, border):
    if type(border) is int:
        border = (border,) * input.ndim

    w_start = input[:, :, :, :1]
    w_start_padding = T.repeat(w_start, border, axis=3)
    w_end = input[:, :, :, -1:]
    w_end_padding = T.repeat(w_end, border, axis=3)

    w_padded = T.concatenate([w_start_padding, input,
                              w_end_padding], axis=3)

    h_start = w_padded[:, :, :1, :]
    h_start_padding = T.repeat(h_start, border, axis=2)
    h_end = w_padded[:, :, -1:, :]
    h_end_padding = T.repeat(h_end, border, axis=2)

    padded = T.concatenate([h_start_padding, w_padded,
                            h_end_padding], axis=2)
    return padded


def gaussian_filter_1d(input, sigma, window_radius=40, axis=-1, border_mode='constant'):
    """
    Filter 1d input with a Gaussian using mode `nearest`.
    input is expected to be three dimensional of type n times x times y
    """
    ndim = 4
    assert input.ndim == ndim, \
        "there must be {} dimensions, got {}".format(ndim, input.ndim)

    dim_pattern = ['x']*input.ndim
    dim_pattern[axis] = 1
    window_radius = gaussian_kernel_default_radius(sigma, window_radius)
    padded_input = add_border_repeat(input, window_radius)
    filter_1d = gaussian_kernel_1d(sigma, window_radius)
    filter_W = filter_1d.dimshuffle(dim_pattern)

    filter_shape = [1] * ndim
    filter_shape[axis] = None
    blur = T.nnet.conv2d(padded_input, filter_W, border_mode='valid',
                         filter_shape=filter_shape)
    return blur


def gaussian_filter_2d(input, sigma, window_radius=None,
                       border_mode='repeat'):
    """
    Filter 1d input with a Gaussian using mode `nearest`.
    input is expected to be three dimensional of type n times x times y
    """
    def dimpattern(axis):
        dim_pattern = ['x']*ndim
        dim_pattern[axis] = 0
        return tuple(dim_pattern)

    ndim = 4
    assert input.ndim == ndim, \
        "there must be {} dimensions, got {}".format(ndim, input.ndim)
    window_radius = gaussian_kernel_default_radius(sigma, window_radius)
    padded_input = add_border(input, window_radius, border_mode)
    filter_1d = gaussian_kernel_1d(sigma, window_radius)
    print(dimpattern(-1))
    filter_w = filter_1d.dimshuffle(dimpattern(-1))
    blur_w = T.nnet.conv2d(padded_input, filter_w, border_mode='valid',
                           filter_shape=[1, 1, 1, None])
    filter_h = filter_1d.dimshuffle(dimpattern(-2))
    blur = T.nnet.conv2d(blur_w, filter_h, border_mode='valid',
                         filter_shape=[1, 1, None, 1])
    return blur


def smooth(input, sigma=2/4, add_border=True, nb_channels=1):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]
    kernel = gaussian_kernel(sigma, nb_channels=nb_channels)
    size = kernel.shape[-1]
    if add_border:
        input = add_border_reflect(input, size//2)
    smooth_layer = Convolution2D(nb_channels, size, size, border_mode='valid',
                                 input_shape=(1, None, None))
    smooth_layer.build()
    smooth_layer.W.set_value(kernel)
    smooth_layer.input = input
    return smooth_layer.get_output(train=False)


class BorderReflect(Layer):
    def __init__(self, border, **kwargs):
        from warnings import warn
        super().__init__(**kwargs)
        warn("Meaning of filter_size has changed")
        self.border = border

    def get_output(self, train=False):
        return add_border_reflect(self.get_input(train), self.border)

    @property
    def output_shape(self):
        shp = self.input_shape
        b = self.border
        return shp[:2] + (shp[2] + 2*b, shp[3] + 2*b)


def add_border_reflect(input, border):
    """Reflects the border like OpenCV BORDER_REFLECT_101. See here
    http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html"""

    shp = input.shape
    wb = T.zeros((shp[0], shp[1], shp[2]+2*border, shp[3]+2*border))
    wb = T.set_subtensor(wb[:, :, border:shp[2]+border, border:shp[3]+border], input)

    top = input[:, :, 1:border+1, :]
    wb = T.set_subtensor(wb[:, :, :border, border:shp[3]+border], top[:, :, ::-1, :])

    bottom = input[:, :, -border-1:-1, :]
    wb = T.set_subtensor(wb[:, :, -border:, border:shp[3]+border], bottom[:, :, ::-1, :])

    left = input[:, :, :, 1:border+1]
    wb = T.set_subtensor(wb[:, :, border:shp[2]+border, :border], left[:, :, :, ::-1])

    right = input[:, :, :, -border-1:-1]
    wb = T.set_subtensor(wb[:, :, border:shp[2]+border, -border:], right[:, :, :, ::-1])

    left_top = input[:, :, 1:border+1, 1:border+1]
    wb = T.set_subtensor(wb[:, :, :border, :border], left_top[:, :, ::-1, ::-1])
    left_bottom = input[:, :, -border-1:-1, 1:border+1]
    wb = T.set_subtensor(wb[:, :, -border:, :border], left_bottom[:, :, ::-1, ::-1])
    right_bottom = input[:, :, 1:border+1, -border-1:-1]
    wb = T.set_subtensor(wb[:, :, :border, -border:], right_bottom[:, :, ::-1, ::-1])
    right_top = input[:, :, -border-1:-1, -border-1:-1]
    wb = T.set_subtensor(wb[:, :, -border:, -border:], right_top[:, :, ::-1, ::-1])
    return wb


def sobel(img, add_border=False):
    conv_x = Convolution2D(1, 3, 3, border_mode='valid',
                           input_shape=(1, None, None))
    conv_y = Convolution2D(1, 3, 3, border_mode='valid',
                           input_shape=(1, None, None))
    filter = cast_to_floatx(np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]))
    if add_border:
        img = add_border_reflect(img, border=3)
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
    input = add_border_reflect(input, border=2)
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
