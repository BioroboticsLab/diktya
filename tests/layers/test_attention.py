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
from unittest.mock import Mock, MagicMock
import math
from seya.data_utils import floatX
import theano

from beras.layers.attention import RotationTransformer
import numpy as np


def test_rotates_images():
    bs = 3
    img = theano.shared(np.zeros((bs, 1, 8, 8), dtype=floatX))

    angle = np.asarray([0, math.pi / 2, math.pi], dtype=floatX)
    angle = theano.shared(angle)

    locnet = Mock()
    locnet.layers = [Mock()]
    locnet.layers[0].input = img
    locnet.get_output = Mock(return_value=angle)
    assert locnet.get_output() == angle
    rot_layer = RotationTransformer(locnet, return_theta=True)
    rot_layer.input = img
    theta = rot_layer.get_output()
    np_theta = theta.eval()
    np.testing.assert_almost_equal(
        np_theta,
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


