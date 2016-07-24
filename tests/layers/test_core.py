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

from diktya.layers.core import Swap, Subtensor, SplitAt, InBounds, BatchLoss
import numpy as np
import theano
from keras.models import Sequential, Model
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
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


def test_subtensor():
    layer = Subtensor(0, 1)
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


def test_in_bounds_clip():
    layer = InBounds(-1, 1, clip=True)
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


def test_in_bounds_regularizer():
    model = Sequential()
    model.add(InBounds(-1, 1, clip=True, input_shape=(1,)))
    model.compile('adam', 'mse')
    assert model.metrics_names == ['loss', 'reg']

    loss, reg = model.train_on_batch(np.array([[0]]), np.array([[0]]))

    assert float(loss) == 0

    loss_on_2, reg = model.train_on_batch(np.array([[2]]), np.array([[1]]))
    assert float(loss_on_2) > 0

    loss_on_100, reg = model.train_on_batch(np.array([[100]]), np.array([[1]]))
    assert float(loss_on_2) < float(loss_on_100)


def test_batch_loss():
    bl = BatchLoss(axis=1, normalize=True)
    shape = (1, 8, 8)
    input = Input(shape=shape)
    conv_out = Convolution2D(4, 3, 3, border_mode='same')(input)
    x = bl(conv_out)
    x = Activation('relu')(x)
    m = Model(input, x)
    m.compile('adam')
    m.fit(np.random.uniform(-1, 1, (10000,) + shape),
          batch_size=128,
          nb_epoch=10)

    m_conv = Model(input, conv_out)
    data = np.random.uniform(-1, 1, (1000,) + shape)
    conv_out = m_conv.predict(data)
    assert np.mean(abs(conv_out.mean(axis=(0, 2, 3)))) <= 0.1
    assert np.mean(np.abs(1 - conv_out.std(axis=(0, 2, 3)))) <= 0.1
