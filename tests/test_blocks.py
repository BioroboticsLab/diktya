# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from diktya.blocks import conv2d_block, resnet
from diktya.func_api_helpers import sequential
from keras.layers import Input
from keras.engine.training import Model


def test_conv2d_block():
    x = Input(shape=(1, 8, 8))
    y = sequential(
        conv2d_block(4)
    )(x)
    model = Model(x, y)
    assert model.get_output_shape_for((None, 1, 8, 8)) == (None, 4, 8, 8)

    x = Input(shape=(1, 8, 8))
    y = sequential(
        conv2d_block(4, pooling='avg')
    )(x)
    model = Model(x, y)
    assert model.get_output_shape_for((None, 1, 8, 8)) == (None, 4, 4, 4)

    x = Input(shape=(1, 8, 8))
    y = sequential(
        conv2d_block(4, up=True)
    )(x)
    model = Model(x, y)
    assert model.get_output_shape_for((None, 1, 8, 8)) == (None, 4, 16, 16)


def test_resnet():
    n = 4
    x = Input(shape=(1, 8, 8))
    y = sequential([
        conv2d_block(n),
        resnet(n)
    ])(x)
    model = Model(x, y)
    assert model.get_output_shape_for((None, 1, 8, 8)) == (None, n, 8, 8)
