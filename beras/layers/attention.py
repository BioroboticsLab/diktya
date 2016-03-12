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
from keras.layers.core import Layer
import theano.tensor as T


class RotationTransformer(SpatialTransformer):
    '''A Spatial Transformer limitted to rotation '''
    def __init__(self, rot_layer, *args, **kwargs):
        self.rot_layer = rot_layer
        fake_model = Sequential()
        fake_model.add(Layer(input_shape=(1, 1, 1)))
        super().__init__(fake_model, *args, **kwargs)

    @property
    def output_shape(self):
        return self.input_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        rot_angle = self.rot_layer.get_output(train)
        rot_angle = rot_angle.reshape((X.shape[0], 1))

        # set up rotation matrix
        cos = T.cos(rot_angle)
        sin = T.sin(rot_angle)
        zeros = T.zeros_like(rot_angle)
        theta = T.concatenate([cos, -sin, zeros, sin, cos, zeros],
                              axis=1).reshape((-1, 2, 3))

        output = self._transform(theta, X, self.downsample_factor)
        if self.return_theta:
            return theta.reshape((X.shape[0], 2, 3))
        else:
            return output


class GraphSpatialTransformer(SpatialTransformer):
    '''A Spatial Transformer limitted to rotation '''
    def __init__(self, layer, *args, **kwargs):
        self.layer = layer
        fake_model = Sequential()
        fake_model.add(Layer(input_shape=(1, 1, 1)))
        super().__init__(fake_model, *args, **kwargs)

    @property
    def output_shape(self):
        return self.input_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        theta = self.layer.get_output(train)
        theta = theta.reshape((X.shape[0], 2, 3))
        return self._transform(theta, X, self.downsample_factor)
