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
import scipy
import skimage
import skimage.transform
import theano
import numpy as np
from beras.util import _add_virtual_border, downsample, upsample
import matplotlib.pyplot as plt


visual_debug = False


def test_add_border():
    filter_size = 7
    half = (filter_size - 1) // 2
    x = theano.shared(np.random.sample((3, 1, 64, 64)))
    x_with_border = _add_virtual_border(x, filter_size).eval()
    v = x.get_value()
    top = v[:, :, 1:half+1, :]
    assert (x_with_border[:, :, half:half+64, half:half+64] == v).all()
    assert (x_with_border[:, :, :half, half:half+64] == top[:, :, ::-1, :]).all()
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

