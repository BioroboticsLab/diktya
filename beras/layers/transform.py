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

from keras.layers.core import Layer
import keras.backend as K
import theano.tensor as T
import numpy as np


class iDCT(Layer):
    def build(self):
        def dct_iii(n):
            mat = np.zeros((n, n), dtype=K.floatx())
            for k in range(n):
                for i in range(n):
                    if i == 0:
                        mat[k, i] = 0.5
                    else:
                        mat[k, i] = np.cos(np.pi/n * i * (k + 0.5))
            return mat
        shp = self.input_shape
        assert shp[-1] == shp[-2], \
            "tensor must be symetric in the last two dimensions"
        self.D = T.constant(dct_iii(shp[2]))

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.dot(K.dot(X, self.D), self.D.transpose())
