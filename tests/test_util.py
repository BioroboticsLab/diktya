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
import random
from timeit import Timer

import pytest
from keras.backend.common import cast_to_floatx
from tests import visual_debug, TEST_OUTPUT_DIR
from colorsys import hsv_to_rgb
import scipy
import skimage
import skimage.transform
import theano
import numpy as np
from beras.util import _add_virtual_border, downsample, upsample, tile, \
    smooth, sobel
import matplotlib.pyplot as plt
import theano.tensor as T
import skimage.data
import skimage.color

@pytest.fixture
def astronaut():
    astronaut = skimage.data.astronaut() / 255.
    astronaut = skimage.color.rgb2gray(astronaut)
    return cast_to_floatx(astronaut[::4, ::4])


def plt_save_and_maybe_show(fname):
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, fname))
    if visual_debug:
        plt.show()


def test_add_border():
    filter_size = 7
    half = (filter_size - 1) // 2
    x = theano.shared(np.random.sample((3, 1, 64, 64)))
    x_with_border = _add_virtual_border(x, filter_size).eval()
    v = x.get_value()
    top = v[:, :, 1:half+1, :]
    np.testing.assert_allclose(
        x_with_border[:, :, half:half+64, half:half+64], v)
    np.testing.assert_allclose(x_with_border[:, :, :half, half:half+64],
                               top[:, :, ::-1, :])
    if visual_debug:
        plt.subplot(121)
        plt.imshow(x.get_value()[2, 0, :])
        plt.subplot(122)
        plt.imshow(x_with_border[2, 0, :])
        plt.show()


def test_benchmark_add_border():
    filter_size = 7
    # x = T.tensor4()
    x = theano.shared(np.random.sample((128, 1, 64, 64)))
    x_with_border = _add_virtual_border(x, filter_size)
    add_border = theano.function([], [x_with_border])
    t = Timer(lambda: add_border())
    n = 500
    print("add_border took: {:.4f}ms".format(1000 * t.timeit(number=n) / n))


def test_sobel(astronaut):
    x = theano.shared(astronaut[np.newaxis, np.newaxis])
    sobel_x, sobel_y = [s.eval() for s in sobel(x)]
    plt.subplot(131)
    plt.imshow(x.get_value()[0, 0, :])
    plt.subplot(132)
    plt.imshow(sobel_x[0, 0, :])
    plt.subplot(133)
    plt.imshow(sobel_y[0, 0, :])
    plt_save_and_maybe_show("test_sobel.png")
    plt.show()


def test_smooth(astronaut):
    x = theano.shared(astronaut[np.newaxis, np.newaxis, :64, :64])
    smoothed = smooth(x, sigma=3).eval()
    assert smoothed.shape == x.get_value().shape
    plt.subplot(121)
    plt.imshow(x.get_value()[0, 0, :])
    plt.subplot(122)
    plt.imshow(smoothed[0, 0, :])
    plt_save_and_maybe_show("test_smooth.png")
    plt.show()


def test_downsample(astronaut):
    astronaut = skimage.transform.resize(astronaut, (64, 64))
    x = theano.shared(astronaut.reshape(1, 1, 64, 64))
    x_small = downsample(x).eval()
    plt.subplot(121)
    plt.imshow(x.get_value()[0, 0, :])
    plt.subplot(122)
    plt.imshow(x_small[0, 0, :])
    plt_save_and_maybe_show("test_downsample.png")


def test_upsample(astronaut):
    x = theano.shared(astronaut[np.newaxis, np.newaxis])
    x_up = upsample(downsample(x)).eval()
    plt.subplot(121)
    plt.imshow(x.get_value()[0, 0, :])
    plt.subplot(122)
    plt.imshow(x_up[0, 0, :])
    plt_save_and_maybe_show("test_upsample.png")


def test_tile():
    n = 128
    images = []

    height, width = 16, 16
    for i in range(n):
        color = hsv_to_rgb(random.random(), 1, 1)
        image = np.zeros((3, 16, 16))
        for c in range(len(color)):
            image[c] = color[c]
        images.append(image)

    number = 20
    tiled = tile(images, columns_must_be_multiple_of=number)

    rows = tiled.shape[1] // height
    cols = tiled.shape[2] // width
    assert cols % number == 0
    for r in range(rows):
        for c in range(cols):
            idx = cols*r + c
            ri = r*height
            ci = c*width
            subimage = tiled[:, ri:ri+height, ci:ci+height]
            if idx < len(images):
                np.testing.assert_allclose(subimage, images[idx])

    fig = plt.figure()
    h_w_rgb = np.zeros((tiled.shape[1], tiled.shape[2], tiled.shape[0]))
    for i in range(3):
        h_w_rgb[:, :, i] = tiled[i]
    plt.imshow(h_w_rgb)
    plt_save_and_maybe_show("test_tile.png")
    plt.close(fig)
