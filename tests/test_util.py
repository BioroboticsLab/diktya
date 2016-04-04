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
from beras.util import add_border_reflect, collect_layers
import matplotlib.pyplot as plt
from conftest import plt_save_and_maybe_show
from keras.layers.core import Dense
from keras.engine.topology import Input, merge
import pytest


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


def test_collect_layers():
    input = Input(shape=(5,))
    layer_a = Dense(20)
    layer_b = Dense(20)
    layer_c = Dense(20)
    layer_d = Dense(20)
    a = layer_a(input)
    b = layer_b(a)
    c = layer_c(b)
    d = layer_d(c)

    layers = collect_layers([b], [d])
    assert layer_c in layers
    assert layer_d in layers
    assert len(layers) == 2

    layers = collect_layers([input], [c])
    assert layer_a in layers
    assert layer_b in layers
    assert layer_c in layers
    assert len(layers) == 3


def test_collect_layers_mimo():
    x = Input(shape=(5,))
    y = Input(shape=(5,))
    layer_a = Dense(20)
    layer_b = Dense(20)
    layer_c = Dense(20)
    layer_d = Dense(20)
    layer_e = Dense(20)
    a = layer_a(x)
    b = layer_b(y)
    m = merge([a, b])
    c = layer_c(m)
    d = layer_d(m)
    e = layer_e(d)

    layers = collect_layers([x, y], [c, e])
    # pytest.set_trace()
    assert layer_a in layers
    assert layer_b in layers
    assert m._keras_history[0] in layers
    assert layer_c in layers
    assert layer_d in layers
    assert layer_e in layers

    layers = collect_layers([x, y], [e])
    assert layer_c not in layers

    # missing inputs are detected
    with pytest.raises(Exception):
        layers = collect_layers([x], [c, e])
