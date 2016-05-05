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
from beras.util import add_border_reflect, collect_layers, sequential, \
    concat, from_config

import matplotlib.pyplot as plt
from conftest import plt_save_and_maybe_show
from keras.layers.core import Dense
from keras.engine.topology import Input, merge
from keras.engine.training import collect_trainable_weights
from keras.models import Model
import pytest


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


def test_concat():
    x = Input(shape=(20,))
    y = Input(shape=(20,))
    model = Model([x, y], concat([x, y]))
    in_x = np.random.sample((32, 20))
    in_y = np.random.sample((32, 20))
    model.compile('adam', 'mse')
    assert model.predict_on_batch([in_x, in_y]).shape == (32, 40)


def test_sequential():
    x = Input(shape=(20,))
    seq = sequential([
        Dense(20),
        Dense(10),
        Dense(1),
    ])

    out = seq(x)
    model = Model([x], [out])
    model.compile('adam', 'mse')
    model.predict_on_batch(np.random.sample((64, 20)))


def test_sequential_flatten():
    x = Input(shape=(20,))
    seq = sequential([
        Dense(20),
        [
            Dense(10)
        ],
        [
            [
                Dense(1)
            ],
            Dense(1)
        ]
    ])
    out = seq(x)
    model = Model([x], [out])
    model.compile('adam', 'mse')
    model.predict_on_batch(np.random.sample((64, 20)))


def test_sequential_trainable():
    x = Input(shape=(20,))
    dense1 = Dense(20)
    dense2 = Dense(10)
    dense3 = Dense(1)
    seq = sequential([
        dense1,
        dense2,
        dense3,
    ], trainable=False)
    seq(x)
    assert collect_trainable_weights(dense1) == []
    assert collect_trainable_weights(dense2) == []
    assert collect_trainable_weights(dense3) == []


def test_sequential_namespace():
    x = Input(shape=(20,))
    dense1 = Dense(20)
    dense2 = Dense(10)
    dense3 = Dense(1)
    seq = sequential([
        dense1,
        dense2,
        dense3,
    ], ns='hello')
    seq(x)
    assert dense1.name.startswith('hello.')
    assert dense2.name.startswith('hello.')
    assert dense3.name.startswith('hello.')


def test_sequential_enumerate():
    x = Input(shape=(20,))
    dense1 = Dense(20)
    dense2 = Dense(10)
    dense3 = Dense(1)
    seq = sequential([
        dense1,
        dense2,
        dense3,
    ], ns='hello')
    seq(x)
    assert dense1.name.endswith('hello.00_dense')
    assert dense2.name.endswith('hello.01_dense')
    assert dense3.name.endswith('hello.02_dense')


def test_from_config_model():
    x = Input(shape=(20,), name='x')
    seq = sequential([
        Dense(20),
        Dense(10),
        Dense(1),
    ], ns='hello')
    model = Model(x, seq(x))
    config = model.get_config()
    ins, out = from_config(config)
    for l in collect_layers(ins, out):
        assert type(l) == Dense


def test_from_config_given_inputs():
    x = Input(shape=(20,), name='input_x')
    seq = sequential([
        Dense(20),
        Dense(10),
        Dense(1),
    ], ns='hello')
    model = Model(x, seq(x))
    config = model.get_config()
    import json
    print(json.dumps(config, indent=2))
    y = Input(shape=(20,), name='input_y')
    ins, out = from_config(config, inputs={'input_x': y})
    assert ins == [y]
    for l in collect_layers(y, out):
        assert type(l) == Dense
