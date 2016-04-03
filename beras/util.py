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

import theano.tensor as T


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


def add_border_reflect(input, border):
    """Reflects the border like OpenCV BORDER_REFLECT_101. See here
    http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html"""

    b = border
    shp = input.shape
    wb = T.zeros((shp[0], shp[1], shp[2]+2*b, shp[3]+2*b))
    wb = T.set_subtensor(wb[:, :, b:shp[2]+b, b:shp[3]+b], input)

    top = input[:, :, 1:b+1, :]
    wb = T.set_subtensor(wb[:, :, :b, b:shp[3]+b], top[:, :, ::-1, :])

    bottom = input[:, :, -b-1:-1, :]
    wb = T.set_subtensor(wb[:, :, -b:, b:shp[3]+b], bottom[:, :, ::-1, :])

    left = input[:, :, :, 1:b+1]
    wb = T.set_subtensor(wb[:, :, b:shp[2]+b, :b], left[:, :, :, ::-1])

    right = input[:, :, :, -b-1:-1]
    wb = T.set_subtensor(wb[:, :, b:shp[2]+b, -b:], right[:, :, :, ::-1])

    left_top = input[:, :, 1:b+1, 1:b+1]
    wb = T.set_subtensor(wb[:, :, :b, :b], left_top[:, :, ::-1, ::-1])
    left_bottom = input[:, :, -b-1:-1, 1:b+1]
    wb = T.set_subtensor(wb[:, :, -b:, :b], left_bottom[:, :, ::-1, ::-1])
    right_bottom = input[:, :, 1:b+1, -b-1:-1]
    wb = T.set_subtensor(wb[:, :, :b, -b:], right_bottom[:, :, ::-1, ::-1])
    right_top = input[:, :, -b-1:-1, -b-1:-1]
    wb = T.set_subtensor(wb[:, :, -b:, -b:], right_top[:, :, ::-1, ::-1])
    return wb
