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
import random
from . import visual_debug, TEST_OUTPUT_DIR
from colorsys import hsv_to_rgb
import scipy
import skimage
import skimage.transform
import theano
import numpy as np
from beras.util import _add_virtual_border, downsample, upsample, tile
import matplotlib.pyplot as plt


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


def test_downsample():
    lena = scipy.misc.lena() / 255
    lena = skimage.transform.resize(lena, (64, 64))
    x = theano.shared(lena.reshape(1, 1, 64, 64))
    x_small = downsample(x).eval()
    if visual_debug:
        plt.subplot(121)
        plt.imshow(x.get_value()[0, 0, :])
        plt.subplot(122)
        plt.imshow(x_small[0, 0, :])
        plt.show()


def test_upsample():
    lena = scipy.misc.lena() / 255
    x = theano.shared(lena.reshape(1, 1, 512, 512))
    x_up = upsample(upsample(downsample(x))).eval()
    if visual_debug:
        plt.subplot(121)
        plt.imshow(x.get_value()[0, 0, :])
        plt.subplot(122)
        plt.imshow(x_up[0, 0, :])
        plt.show()


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

    tiled = tile(images)

    rows = tiled.shape[1] // height
    cols = tiled.shape[2] // width
    for r in range(rows):
        for c in range(cols):
            idx = cols*r + c
            ri = r*height
            ci = c*width
            subimage = tiled[:, ri:ri+height, ci:ci+height]
            if idx < len(images):
                np.testing.assert_allclose(subimage, images[idx])

    fig = plt.figure()
    plt.imshow(tiled.transpose((1, 2, 0)))
    plt.savefig(TEST_OUTPUT_DIR + "/test_tile.png")
    if visual_debug:
        plt.show()
    plt.close(fig)
