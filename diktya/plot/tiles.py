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


def visualise_tiles(images, show=True):
    """

    """
    tiled_fakes = tile(images)
    if len(tiled_fakes) == 3:
        plt.imshow(np.moveaxis(tiled_fakes, 0, -1))
    elif len(tiled_fakes) == 1:
        plt.imshow(tiled_fakes[0], cmap='gray')
    else:
        raise Exception("Expected color or gray image.")

    plt.grid(False)
    plt.axis('off')
    if show:
        plt.show()
