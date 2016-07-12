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

import numpy as np
from beras.util import sequential, concat

from keras.layers.core import Dense
from keras.engine.topology import Input
from keras.engine.training import collect_trainable_weights
from keras.models import Model


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
