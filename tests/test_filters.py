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
import skimage
import skimage.transform
import theano
import numpy as np
from beras.filters import sobel, gaussian_filter_2d, \
    gaussian_filter_2d_variable_sigma, gaussian_kernel_1d
import matplotlib.pyplot as plt
import skimage.data
import skimage.color
import skimage.filters
from conftest import plt_save_and_maybe_show


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

    def test_border_mode(mode, skimage_mode):
        img = theano.shared(astronaut_stacked[np.newaxis])
        sigma = 3
        theano_blur = gaussian_filter_2d(img, sigma, border_mode=mode)
        blur = theano_blur.eval()
        assert blur.shape == (1, 2, 64, 64)
        blur = blur[0]
        expected = skimage.filters.gaussian_filter(
            astronaut, sigma, mode=skimage_mode)
        expected_transposed = skimage.filters.gaussian_filter(
            astronaut.T, sigma, mode=skimage_mode)
        np.testing.assert_allclose(blur[0], expected, rtol=0.001, atol=0.002)
        np.testing.assert_allclose(blur[1], expected_transposed,
                                   rtol=0.001, atol=0.002)
        plt.subplot(221)
        plt.imshow(blur[0], cmap='gray')
        plt.subplot(222)
        plt.imshow(expected, cmap='gray')

        plt.subplot(223)
        plt.imshow(blur[1], cmap='gray')
        plt.subplot(224)
        plt.imshow(expected_transposed, cmap='gray')

        plt_save_and_maybe_show("test_gaussian_blur_2d_{}.png".format(mode))

    test_border_mode('zero', 'constant')


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
    theano_blur = gaussian_filter_2d_variable_sigma(img, sigmas_shared,
                                                    border_mode='zero')
    blur = theano_blur.eval()
    assert blur.shape == (bs, 1, 64, 64)
    blur = blur.reshape(bs, 64, 64)
    r, c = 4, 2
    for i, (sigma, astro) in enumerate(zip(sigmas, astronaut_stacked)):
        expected = skimage.filters.gaussian_filter(astro,
                                                   float(sigma), mode='constant')

        np.testing.assert_allclose(blur[i], expected, rtol=0.01, atol=0.02)

        plt.subplot(r, c, 2*i+1)
        plt.imshow(blur[0], cmap='gray')
        plt.subplot(r, c, 2*(i+1))
        plt.imshow(expected, cmap='gray')

    plt_save_and_maybe_show("test_gaussian_blur_2d_variable_sigmas.png")
