# Copyright 2015 Leon Sixt
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

from seya.layers.attention import SpatialTransformer
from keras.models import Sequential
from keras.layers.core import InputSpec
from keras.engine.topology import InputLayer
import theano.tensor as T


class RotationTransformer(SpatialTransformer):
    '''A Spatial Transformer limitted to rotation '''
    def __init__(self, *args, **kwargs):
        fake_model = Sequential()
        fake_model.add(InputLayer(input_shape=(1, 1, 1)))
        super().__init__(fake_model, *args, **kwargs)

    def get_output_shape_for(self, input_shapes):
        x_input_shape, rot_input_shape = input_shapes
        return [x_input_shape, rot_input_shape[:1] + (6,)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=2)]

    def compute_mask(self, input, mask=None):
        return [None, None]

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'RotationTransformer must be called with two tensors')

        x, rot_angle = inputs

        # set up rotation matrix
        cos = T.cos(rot_angle)
        sin = T.sin(rot_angle)
        zeros = T.zeros_like(rot_angle)
        theta = T.concatenate([cos, -sin, zeros, sin, cos, zeros],
                              axis=1).reshape((-1, 2, 3))

        output = self._transform(theta, x, self.downsample_factor)
        return [output, theta.reshape((x.shape[0], 6))]
