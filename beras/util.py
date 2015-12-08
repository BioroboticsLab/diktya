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
from keras.utils.theano_utils import floatX
from scipy.misc import imsave


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
]]], dtype=theano.config.floatX)


def upsample(input):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]

    upsample_layer = Convolution2D(1, 5, 5, border_mode='valid',
                                   input_shape=(1, None, None))
    upsample_layer_weight = _gaussian_blur_kernel/64.
    upsample_layer.W = theano.shared(upsample_layer_weight)
    shp = input.shape
    upsampled = T.zeros((shp[0], shp[1], 2*shp[2], 2*shp[3]))
    upsampled = T.set_subtensor(upsampled[:, :, ::2, ::2], input)
    upsampled = _add_virtual_border(upsampled)
    upsample_layer.input = upsampled
    return upsample_layer.get_output(train=False)


def blur(input, sigma=2., add_border=True):
    import skimage.filters
    if type(input) == list:
        assert len(input) == 1
        input = input[0]

    size = ceil(2*sigma+1)
    a = np.zeros((size, size))
    center = size // 2
    a[center, center] = 1
    gaussian_kernel = skimage.filters.gaussian_filter(a, sigma)
    gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis]
    if add_border:
        input = _add_virtual_border(input, filter_size=size)
    blur_layer = Convolution2D(1, size, size, border_mode='valid',
                               input_shape=(1, None, None))
    blur_layer.W = theano.shared(floatX(gaussian_kernel))
    blur_layer.input = input
    return blur_layer.get_output(train=False)


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
    filter = floatX(np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]))
    conv_x.W = theano.shared(filter[np.newaxis, np.newaxis])
    conv_y.W = theano.shared(np.transpose(filter)[np.newaxis, np.newaxis])
    conv_x.input = img
    conv_y.input = img
    return conv_x.get_output(train=False), conv_y.get_output(train=False)


def downsample(input):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]

    downsample_layer = Convolution2D(1, 5, 5, subsample=(2, 2),
                                      border_mode='valid',
                                      input_shape=(1, None, None))
    downsample_layer_weight = _gaussian_blur_kernel/256.
    downsample_layer.W = theano.shared(downsample_layer_weight)
    input = _add_virtual_border(input)
    downsample_layer.input = input
    return downsample_layer.get_output(train=False)


def tile(tiles):
    def calc_columns_rows(n):
        num_columns = int(ceil(sqrt(n)))
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
