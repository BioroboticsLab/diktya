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


class SplitAt(Layer):
    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.input_spec = [InputSpec(dtype='int32'),
                           InputSpec(dtype=K.floatx())]

    def get_output_shape_for(self, input_shapes):
        shp = input_shapes[1]
        new_shp = list(shp)
        new_shp[self.axis] = None
        return [tuple(new_shp), tuple(new_shp)]

    def compute_mask(self, x, masks=None):
        return [None, None]

    def call(self, xs, mask=None):
        def build_index(start, stop):
            index = []
            for i in range(K.ndim(arr)):
                if i == self.axis:
                    index.append(slice(start, stop, 1))
                else:
                    index.append(slice(None, None, 1))
            return index
        idx, arr = xs
        idx = idx.flatten()[0]
        front_index = build_index(0, idx)
        back_index = build_index(idx, None)
        return [arr[tuple(front_index)], arr[tuple(back_index)]]


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


class Switch(Layer):
    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        return input_shape[1]

    def call(self, x, mask=None):
        condition, then_expr, else_expr = x
        pattern = (0, 1) + ('x',) * (K.ndim(then_expr) - 2)
        return K.switch(condition.dimshuffle(*pattern), then_expr, else_expr)


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
