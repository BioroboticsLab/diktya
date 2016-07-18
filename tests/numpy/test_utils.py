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

from colorsys import hsv_to_rgb
import numpy as np
from diktya.numpy.utils import tile
import matplotlib.pyplot as plt
from conftest import plt_save_and_maybe_show


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
