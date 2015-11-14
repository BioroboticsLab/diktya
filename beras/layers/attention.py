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
from seya.data_utils import floatX
from seya.utils import apply_model
import theano.tensor as T


class RotationTransformer(SpatialTransformer):
    '''A Spatial Transformer limitted to rotation '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        rot_angle = apply_model(self.locnet, X, train)
        rot_angle = rot_angle.reshape((X.shape[0], 1))

        # set up rotation matrix
        cos = T.cos(rot_angle)
        sin = T.sin(rot_angle)
        zeros = T.zeros_like(rot_angle)
        theta = T.concatenate([cos, -sin, zeros, sin, cos, zeros], axis=1).reshape((-1, 2, 3))

        output = self._transform(theta, X, self.downsample_factor)
        if self.return_theta:
            return theta.reshape((X.shape[0], 2, 3))
        else:
            return output
