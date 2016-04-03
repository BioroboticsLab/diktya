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

import theano
import theano.tensor as T
from beras.util import add_border

from keras.backend.common import floatx, cast_to_floatx
import numpy as np
import keras.backend as K


def sobel(img, border_mode='reflect'):
    filter = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ], dtype=floatx())
    if add_border:
        img = add_border(img, border=1, mode=border_mode)
    kernel_x = theano.shared(filter[np.newaxis, np.newaxis])
    kernel_y = theano.shared(np.transpose(filter)[np.newaxis, np.newaxis])
    conv_x = K.conv2d(img, kernel_x, border_mode='valid',
                      image_shape=(None, 1, None, None),
                      filter_shape=(1, 1, 3, 3))
    conv_y = K.conv2d(img, kernel_y, border_mode='valid',
                      image_shape=(None, 1, None, None),
                      filter_shape=(1, 1, 3, 3))
    return conv_x, conv_y


def gaussian_kernel_default_radius(sigma, window_radius=None):
    if window_radius is None:
        radius = T.cast(T.max(T.ceil(3*sigma)), 'int32')
        if type(sigma) in (float, int):
            return int(radius.eval())
        else:
            return radius
    else:
        return window_radius


def np_gaussian_kernel_1d(sigma, window_radius=None):
    assert type(sigma) in (float, int)
    if window_radius is None:
        window_radius = int(np.ceil(3*sigma))
    index = np.arange(2*window_radius + 1) - window_radius
    kernel = np.exp(-0.5*index**2/sigma**2)
    kernel = kernel / kernel.sum()
    return cast_to_floatx(kernel)


def np_gaussian_kernel_2d(sigma, window_radius=None):
    kernel = np_gaussian_kernel_1d(sigma, window_radius)
    kernel_2d = np.dot(kernel.reshape(-1, 1), kernel.reshape(1, -1))
    assert len(kernel_2d.shape) == 2
    return kernel_2d


def gaussian_kernel_1d(sigma, window_radius=None):
    if type(sigma) in (float, int):
        sigma = T.as_tensor_variable(sigma)
    if sigma.ndim == 0:
        sigma = sigma.reshape((1,))
    nb_sigmas = sigma.shape[0]
    sigma = sigma.reshape((nb_sigmas, 1))
    radius = gaussian_kernel_default_radius(sigma, window_radius)
    index = T.arange(2*radius + 1) - radius
    index = T.cast(index, 'float32')
    index = T.tile(index, nb_sigmas).reshape((nb_sigmas, 2*radius+1))
    kernel = T.exp(-0.5*index**2/sigma**2)
    kernel = kernel / kernel.sum(axis=1).dimshuffle(0, 'x')
    kernel = T.cast(kernel, 'float32')
    if type(sigma) in (float, int) and \
            type(window_radius) in (float, int, type(None)):
        evaluated_kernel = T.as_cuda_or_tensor_variable(kernel.eval())
        return evaluated_kernel
    return kernel


def gaussian_filter_2d(input, sigma, window_radius=None,
                       border_mode='reflect'):
    bs, ch, h, w = input.shape
    ndim = 4
    assert input.ndim == ndim, \
        "there must be {} dimensions, got {}".format(ndim, input.ndim)
    window_radius = gaussian_kernel_default_radius(sigma, window_radius)
    input = input.reshape((bs*ch, 1, h, w))
    padded_input = add_border(input, window_radius, border_mode)
    filter_1d = gaussian_kernel_1d(sigma, window_radius)
    dimpattern_w = ('x', 'x', 0, 1)
    dimpattern_h = ('x', 'x', 1, 0)
    filter_w = filter_1d.dimshuffle(dimpattern_w)
    blur_w = T.nnet.conv2d(padded_input,
                           filter_w, border_mode='valid',
                           filter_shape=[1, 1, 1, None])
    filter_h = filter_1d.dimshuffle(dimpattern_h)
    blured = T.nnet.conv2d(blur_w, filter_h, border_mode='valid',
                           filter_shape=[1, 1, None, 1])
    return blured.reshape((bs, ch, h, w))


def gaussian_filter_2d_variable_sigma(input, sigmas, window_radius=None):
    def filter_sigma(idx, kernel):
        dimpattern_w = ('x', 'x', 'x', 0)
        dimpattern_h = ('x', 'x', 0, 'x')
        filter_w = kernel.dimshuffle(dimpattern_w)
        blur_w = T.nnet.conv2d(padded_input[idx:idx+1],
                               filter_w, border_mode='valid',
                               filter_shape=[1, 1, 1, None])
        filter_h = kernel.dimshuffle(dimpattern_h)
        return T.nnet.conv2d(blur_w, filter_h, border_mode='valid',
                             filter_shape=[1, 1, None, 1])
    ndim = 4
    border_mode = 'reflect'
    assert input.ndim == ndim, \
        "there must be {} dimensions, got {}".format(ndim, input.ndim)
    window_radius = gaussian_kernel_default_radius(sigmas, window_radius)
    padded_input = add_border(input, window_radius, border_mode)
    kernel = gaussian_kernel_1d(sigmas, window_radius)
    blur, _ = theano.map(
        filter_sigma,
        sequences=[T.arange(sigmas.shape[0]), kernel])
    return blur.reshape(input.shape)
