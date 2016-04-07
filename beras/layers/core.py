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

from keras.layers.core import Layer, InputSpec
import keras.backend as K
import theano.tensor as T
import theano
from beras.regularizers import ActivityInBoundsRegularizer


class Split(Layer):
    def __init__(self, start, stop, step=1, axis=0, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shp = input_shape
        new_shp = []
        length = (self.stop - self.start) // abs(self.step)
        for i in range(len(shp)):
            if i == self.axis:
                new_shp.append(length)
            else:
                new_shp.append(shp[i])
        return tuple(new_shp)

    def call(self, x, mask=None):
        index = []
        for i in range(K.ndim(x)):
            if i == self.axis:
                index.append(slice(self.start, self.stop, self.step))
            else:
                index.append(slice(None, None, 1))
        return x[tuple(index)]


class Swap(Layer):
    def __init__(self, a, b, **kwargs):
        self.a = a
        self.b = b
        super(Swap, self).__init__(**kwargs)

    def call(self, x, mask=None):
        tmp = x[:, self.a]
        x = T.set_subtensor(x[:, self.a], x[:, self.b])
        x = T.set_subtensor(x[:, self.b], tmp)
        return x


class ZeroGradient(Layer):
    def call(self, x, mask=None):
        return theano.gradient.zero_grad(x)


class LinearInBounds(Layer):
    def __init__(self, low=-1, high=1, clip=False, **kwargs):
        self.activity_in_bounds = ActivityInBoundsRegularizer(low, high)
        self.clip = clip
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None,) + input_shape[1:])]
        self.activity_in_bounds.set_layer(self)
        self.regularizers = [self.activity_in_bounds]
        super().build(input_shape)

    def call(self, x, mask=None):
        if self.clip:
            return T.clip(x, self.activity_in_bounds.low,
                          self.activity_in_bounds.high)
        else:
            return x
