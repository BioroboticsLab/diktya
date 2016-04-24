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

from beras.transform import tile
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def zip_visualise_tiles(*arrs, show=True):
    import matplotlib.pyplot as plt
    assert len(arrs) >= 2
    length = len(arrs[0])
    for a in arrs:
        assert len(a) == length, "all input arrays must have the same size"
    tiles = []
    for i in range(length):
        for a in arrs:
            tiles.append(a[i])

    tiled = tile(tiles, columns_must_be_multiple_of=len(arrs))
    assert len(tiled) == 1, "currently only grayscale image are supported"
    plt.imshow(tiled[0], cmap='gray')
    if show:
        plt.show()


def visualise_tiles(images, show=True):
    import matplotlib.pyplot as plt
    tiled_fakes = tile(images)
    plt.imshow(tiled_fakes[0], cmap='gray')
    if show:
        plt.show()


def _save_image_worker(data, fname, cmap):
    # matplotlib seems to leak memory. Quite a hack
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close(fig)


def save_image(data, fname, cmap='gray'):
    proc = multiprocessing.Process(target=_save_image_worker,
                                   args=(data, fname, cmap))
    proc.daemon = True
    proc.start()
