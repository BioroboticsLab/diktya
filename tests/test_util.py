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
from timeit import Timer

import pytest
from keras.backend.common import cast_to_floatx
from colorsys import hsv_to_rgb
import skimage
import skimage.transform
import theano
import numpy as np
from beras.util import add_border_reflect, downsample, upsample, tile, \
    smooth, sobel, gaussian_filter_2d, gaussian_filter_2d_variable_sigma, \
    gaussian_kernel_1d
import matplotlib.pyplot as plt
import skimage.data
import skimage.color
import skimage.filters
from conftest import plt_save_and_maybe_show


@pytest.fixture
def astronaut():
    astronaut = skimage.data.astronaut() / 255.
    astronaut = skimage.color.rgb2gray(astronaut)
    return cast_to_floatx(astronaut[::4, ::4])



def test_add_border_reflect():
    filter_radius = 3
    r = filter_radius
    x = theano.shared(np.random.sample((3, 1, 64, 64)))
    x_with_border = add_border_reflect(x, filter_radius).eval()
    v = x.get_value()
    top = v[:, :, 1:r+1, :]

    plt.subplot(121)
    plt.imshow(x.get_value()[2, 0, :])
    plt.subplot(122)
    plt.imshow(x_with_border[2, 0, :])
    plt_save_and_maybe_show("add_border_reflect.png")

    np.testing.assert_allclose(
        x_with_border[:, :, r:r+64, r:r+64], v)
    np.testing.assert_allclose(x_with_border[:, :, :r, r:r+64],
                               top[:, :, ::-1, :])

def test_benchmark_add_border():
    filter_size = 7
    # x = T.tensor4()
    x = theano.shared(np.random.sample((128, 1, 64, 64)))
    x_with_border = add_border_reflect(x, filter_size)
    add_border = theano.function([], [x_with_border])
    t = Timer(lambda: add_border())
    n = 500
    print("add_border took: {:.4f}ms".format(1000 * t.timeit(number=n) / n))


def test_sobel(astronaut):
    x = theano.shared(astronaut[np.newaxis, np.newaxis].astype(np.float32))
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


def test_gaussian_kernel_1d():
    def to_theano(x):
        return theano.shared(np.array(x, dtype=np.float32))

    kernel = gaussian_kernel_1d(to_theano([3]), window_radius=10)
    assert kernel.eval().shape == (1, 21)

    kernel = gaussian_kernel_1d(to_theano([3, 2]), window_radius=10)
    assert kernel.eval().shape == (2, 21)


def test_gaussian_filter_2d(astronaut):
    astronaut = astronaut[::2, ::2]
    astronaut_stacked = np.stack([astronaut, astronaut.T])
    img = theano.shared(astronaut_stacked[np.newaxis])
    sigma = 3
    theano_blur = gaussian_filter_2d(img, sigma)
    blur = theano_blur.eval()
    assert blur.shape == (1, 2, 64, 64)
    blur = blur[0]
    expected = skimage.filters.gaussian_filter(astronaut, sigma,
                                               mode='mirror')
    expected_transposed = skimage.filters.gaussian_filter(astronaut.T, sigma,
                                                          mode='mirror')
    np.testing.assert_allclose(blur[0], expected, rtol=0.01, atol=0.02)
    np.testing.assert_allclose(blur[1], expected_transposed,
                               rtol=0.01, atol=0.02)
    plt.subplot(221)
    plt.imshow(blur[0], cmap='gray')
    plt.subplot(222)
    plt.imshow(expected, cmap='gray')

    plt.subplot(223)
    plt.imshow(blur[1], cmap='gray')
    plt.subplot(224)
    plt.imshow(expected_transposed, cmap='gray')

    plt_save_and_maybe_show("test_gaussian_blur_2d.png")


def test_gaussian_filter_2d_variable_sigma(astronaut):
    astronaut = astronaut[::2, ::2]
    astronaut_stacked = np.stack([astronaut, astronaut.T])
    astronaut_stacked = np.stack([astronaut,
                                  astronaut[:, ::-1],
                                  astronaut[::-1, :],
                                  astronaut[::-1, ::-1]])
    bs = len(astronaut_stacked)
    img = theano.shared(astronaut_stacked[:, np.newaxis])
    sigmas = np.array([3, 1, 2, 0.5], dtype=np.float32)
    sigmas_shared = theano.shared(sigmas)
    theano_blur = gaussian_filter_2d_variable_sigma(img, sigmas_shared)
    blur = theano_blur.eval()
    assert blur.shape == (bs, 1, 64, 64)
    blur = blur.reshape(bs, 64, 64)
    r, c = 4, 2
    for i, (sigma, astro) in enumerate(zip(sigmas, astronaut_stacked)):
        expected = skimage.filters.gaussian_filter(astro,
                                                   float(sigma), mode='mirror')

        np.testing.assert_allclose(blur[i], expected, rtol=0.01, atol=0.02)

        plt.subplot(r, c, 2*i+1)
        plt.imshow(blur[0], cmap='gray')
        plt.subplot(r, c, 2*(i+1))
        plt.imshow(expected, cmap='gray')

    plt_save_and_maybe_show("test_gaussian_blur_2d_variable_sigmas.png")


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
