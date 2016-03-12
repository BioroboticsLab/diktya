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


class Split(Layer):
    def __init__(self, start, stop, step=1, axis=0, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.axis = axis
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

    @property
    def output_shape(self):
        shp = self.input_shape
        new_shp = []
        length = (self.stop - self.start) / abs(self.step)
        for i in range(len(shp)):
            if i == self.axis:
                new_shp.append(length)
            else:
                new_shp.append(shp[i])
        return new_shp

    def get_output(self, train=False):
        X = self.get_input(train)
        index = []
        for i in range(K.ndim(X)):
            if i == self.axis:
                index.append(slice(self.start, self.stop, self.step))
            else:
                index.append(slice(None, None, 1))
        return X[tuple(index)]


class Swap(Layer):
    def __init__(self, a, b, **kwargs):
        super(Swap, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

    def get_output(self, train=False):
        X = self.get_input(train)
        tmp = X[:, self.a]
        X = T.set_subtensor(X[:, self.a], X[:, self.b])
        X = T.set_subtensor(X[:, self.b], tmp)
        return X
