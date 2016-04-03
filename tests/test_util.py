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
from timeit import Timer

import theano
import numpy as np
from beras.util import add_border_reflect
import matplotlib.pyplot as plt
from conftest import plt_save_and_maybe_show


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
