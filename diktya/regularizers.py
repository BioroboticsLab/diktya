# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from keras.regularizers import Regularizer
import keras.backend as K
import warnings


class ActivityInBoundsRegularizer(Regularizer):
    def __init__(self, low=-1, high=1, weight=10):
        warnings.warn("ActivityInBoundsRegularizer is deprecated. "
                      "Use the LinearInBounds layer.")
        self.low = low
        self.high = high
        self.weight = weight
        self.uses_learning_phase = True

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityInBoundsRegularizer instance '
                            'before calling the instance.')

        activation = self.layer.input
        l = K.switch(activation < self.low, K.abs(activation - self.low), 0)
        h = K.switch(activation > self.high, K.abs(activation - self.high), 0)
        return K.in_train_phase(loss + self.weight*K.mean(h + l), loss)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "low": self.low,
            "high": self.high
        }


class WeightOrthRegularizer(Regularizer):
    def __init__(self, weight=0.):
        self.weight = K.cast_to_floatx(weight)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightOrthRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')
        regularized_loss = loss
        if self.weight:
            p_flat = K.reshape(self.p, (self.p.shape[0], -1))
            p_length = K.sqrt(K.sum(K.square(p_flat), axis=1) + K.epsilon())
            p_normed = p_flat / (p_length.reshape((p_length.shape[0], 1)) + K.epsilon())
            dist = K.dot(p_normed, K.transpose(p_normed))
            dist *= (1 - K.eye(K.eval(self.p.shape[0])))
            regularized_loss += K.mean(K.abs(dist))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'weight': float(self.weight)}


class SumOfActivityBelowRegularizer(Regularizer):
    def __init__(self, max_sum):
        warnings.warn("SumOfActivityBelowRegularizer is deprecated.")
        # TODO: Write a replacement that uses compute_loss
        self.max_sum = K.variable(max_sum)
        self.uses_learning_phase = True

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception(
                'Need to call `set_layer` on SumOfActivityBelowRegularizer '
                'instance before calling the instance.')

        activation = self.layer.get_output(True)
        axes = (i for i in range(1, len(self.layer.output_shape)))
        sum = K.sum(activation, axis=axes)
        too_big = K.switch(sum > self.max_sum, K.abs(self.max_sum - sum), 0)
        return K.in_train_phase(loss + K.sum(too_big), loss)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "max_sum": self.max_sum,
        }


def with_regularizer(layer, regularizer):
    regularizer.set_layer(layer)
    layer.regularizers.append(regularizer)
    return layer
