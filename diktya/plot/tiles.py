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

from diktya.numpy.utils import tile, zip_tile
import matplotlib.pyplot as plt
import numpy as np


def zip_visualise_tiles(*arrs, show=True):
    tiled = zip_tile(*arrs)
    plt.imshow(tiled, cmap='gray')
    if show:
        plt.show()


def plt_imshow(image, axis=None):
    """
    Plot ``image`` on ``axis``. Can handle color images and gray images of
    shapes ``(3, h, w)``, ``(1, h, w)``, or  ``(h, w)`` where ``h`` = height
    and ``w`` = width.

    If ``axis`` is ``None`` then ``matplotlib.pyplot`` is used.
    """
    if axis is None:
        axis = plt
    if len(image.shape) == 3 and len(image) == 3:  # (3, h, w)
        axis.imshow(np.moveaxis(image, 0, -1))
    elif len(image.shape) == 3 and len(image) == 1:  # (1, h, w)
        axis.imshow(image[0], cmap='gray')
    elif len(image.shape) == 2:   # (h, w)
        axis.imshow(image, cmap='gray')
    else:
        raise Exception("Expected color or gray image. Got image of shape {}"
                        .format(image.shape))


def visualise_tiles(images, show=True):
    """

    """
    tiled_fakes = tile(images)
    plt_imshow(tiled_fakes)
    plt.grid(False)
    plt.axis('off')
    if show:
        plt.show()
