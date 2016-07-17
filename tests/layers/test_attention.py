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
import os
import shutil
import tempfile
from unittest.mock import Mock
import math
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.engine.topology import Input
from keras.models import Sequential, Model
import keras.backend as K
from diktyo.layers.attention import RotationTransformer
import numpy as np


def test_rotates_images():
    bs = 3
    shape = (1, 8, 8)
    img = np.zeros((bs, 1, 8, 8), dtype=K.floatx())
    angle = np.asarray([0, math.pi / 2, math.pi], dtype=K.floatx())

    img_input = Input(shape=shape)
    rot_input = Input(shape=(1,))
    rot_layer = RotationTransformer()([img_input, rot_input])
    model = Model(input=[img_input, rot_input], output=rot_layer)

    model.compile('adam', 'mse')
    _, theta = model.predict([img, angle])
    np.testing.assert_almost_equal(
        theta.reshape(-1, 2, 3),
        np.asarray([
            [[1, 0, 0],
             [0, 1, 0]],

            [[0, -1, 0],
             [1, 0, 0]],
            [[-1, 0, 0],
             [0, -1, 0]],
        ]),
        verbose=True
    )
