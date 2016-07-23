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



class Subtensor(Layer):
    """
    Selects only a part of the input.

    Args:
        start (int): Start index
        stop (int): Stop index
        axis (int): Index along this axis
    """
    def __init__(self, start, stop, step=1, axis=0, **kwargs):
        super(Subtensor, self).__init__(**kwargs)
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

    def get_config(self):
        config = {
            'start': self.start,
            'stop': self.stop,
            'step': self.step,
            'axis': self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    def get_config(self):
        config = {
            'a': self.a,
            'b': self.b,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
    """
    Consider the gradient allways zero.
    Wraps the ``theano.gradient.zero_grad`` function.
    """
    def call(self, x, mask=None):
        return theano.gradient.zero_grad(x)


class InBounds(Layer):
    """
    Between ``low`` and ``high`` this layer is the identity.
    If the value is not in bounds a regularization loss is added to
    the model.

    Args:
        low: lower bound
        high: upper bound
        clip: Clip output if out of bounds
        weight: The regularization loss is multiplied by this

    """
    def __init__(self, low=-1, high=1, clip=False, weight=1, **kwargs):
        self.low = low
        self.high = high
        self.clip = clip
        self.weight = weight
        self.uses_learning_phase = True
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None,) + input_shape[1:])]
        super().build(input_shape)

    def compute_loss(self, input, output, input_mask=None, output_mask=None):
        l = K.switch(input < self.low, K.abs(input - self.low), 0)
        h = K.switch(input > self.high, K.abs(input - self.high), 0)
        return K.in_train_phase(self.weight*K.mean(h + l), 0)

    def call(self, x, mask=None):
        if self.clip:
            return T.clip(x, self.low, self.high)
        else:
            return x

    def get_config(self):
        config = {
            'low': self.low,
            'high': self.high,
            'clip': self.clip,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchLoss(Layer):
    """
    Regularizes the activation to have ``std = 1`` and ``mean = 0``.

    Args:
        axis (int): Axis along to compute the std and mean.
        normalize (bool): Normalize the output by the std and mean of the batch during training.
        weight (float): Weight of the regularization loss
    """
    def __init__(self, axis=1, normalize=True,
                 weight=5.,
                 **kwargs):
        self.axis = axis
        self.normalize = normalize
        self.weight = weight
        super(BatchLoss, self).__init__(**kwargs)

    def compute_loss(self, input, output, input_mask=None, output_mask=None):
        mean, std = self._get_mean_and_std(input)
        return self.weight * (K.abs(mean) + K.abs(std - 1))

    def _get_mean_and_std(self, x):
        reduction_axes = list(range(K.ndim(x)))
        del reduction_axes[self.axis]
        mean = K.mean(x, axis=reduction_axes)
        std = K.std(x, axis=reduction_axes)
        return mean, std

    def call(self, x, mask=None):
        if self.normalize:
            mean, std = self._get_mean_and_std(x)
            broadcast_shape = [1] * K.ndim(x)
            broadcast_shape[self.axis] = K.shape(x)[self.axis]
            broadcast_mean = K.reshape(mean, broadcast_shape)
            broadcast_std = K.reshape(std, broadcast_shape)
            return K.in_train_phase((x - broadcast_mean) / broadcast_std, x)
        else:
            return x

    def get_config(self):
        config = {
            'axis': self.axis,
            'normalize': self.normalize,
            'weight': self.weight,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
