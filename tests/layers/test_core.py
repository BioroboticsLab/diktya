# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from beras.layers.core import Swap, Split, SplitAt, LinearInBounds
import numpy as np
import theano
from keras.models import Sequential, Model
from keras.engine.topology import Input


def test_swap():
    layer = Swap(0, 10)
    shape = (4, 32)
    layer.build(shape)
    arr = np.random.sample(shape).astype(np.float32)
    input = theano.shared(arr)
    output = layer(input).eval()
    swaped = arr.copy()
    swaped[:, 0], swaped[:, 10] = swaped[:, 10].copy(), swaped[:, 0].copy()
    assert (output == swaped).all()


def test_split():
    layer = Split(0, 1)
    shape = (4, 32)
    layer.build(shape)
    arr = np.random.sample(shape).astype(np.float32)
    input = theano.shared(arr)
    output = layer(input).eval()
    assert (output == arr[0:1]).all()


def test_split_at():
    shape = (4, 32)
    idx_shape = (1,)
    arr = np.random.sample(shape).astype(np.float32)
    x = theano.shared(arr)
    bound = 2
    idx = theano.shared(np.cast['int32'](bound))

    split_at = SplitAt(axis=0)
    split_at.build([shape, idx_shape])

    front, back = [o.eval() for o in split_at([idx, x])]
    assert (front == arr[:bound]).all()
    assert (back == arr[bound:]).all()


def test_linear_in_bounds_clip():
    layer = LinearInBounds(-1, 1, clip=True)
    shape = (1, 1)
    layer.build(shape)
    arr = np.array([[0]], dtype=np.float32)
    output = layer(theano.shared(arr)).eval()
    assert (output == arr).all()

    arr = np.array([[0]], dtype=np.float32)
    output = layer(theano.shared(arr)).eval()
    assert (output == arr).all()

    arr = np.array([[2]], dtype=np.float32)
    output = layer(theano.shared(arr)).eval()
    assert float(output) == 1.


def test_linear_in_bounds_regularizer():
    model = Sequential()
    model.add(LinearInBounds(-1, 1, clip=True, input_shape=(1,)))
    model.compile('adam', 'mse')
    loss = model.train_on_batch(np.array([[0]]), np.array([[0]]))
    assert float(loss) == 0

    loss_on_2 = model.train_on_batch(np.array([[2]]), np.array([[1]]))
    assert float(loss_on_2) > 0

    loss_on_100 = model.train_on_batch(np.array([[100]]), np.array([[1]]))
    assert float(loss_on_2) < float(loss_on_100)
