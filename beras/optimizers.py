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


from keras.optimizers import Optimizer
import theano.tensor as T
import keras.backend as K
import numpy as np


class RProp(Optimizer):
    def __init__(self,
                 init_step=0.01,
                 increase=1.10,
                 decrease=0.8,
                 min_step=1e-7,
                 max_step=10,
                 **kwargs):
        self.init_step = init_step
        self.increase = K.variable(increase)
        self.decrease = K.variable(decrease)
        self.min_step = K.variable(min_step)
        self.max_step = K.variable(max_step)
        super(RProp, self).__init__(**kwargs)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        for param, grad, c in zip(params, grads, constraints):
            grad_tm1 = K.variable(np.zeros(K.get_value(param).shape))
            step_tm1 = K.variable(
                self.init_step*np.ones(K.get_value(param).shape))
            test = grad * grad_tm1
            diff = T.lt(test, 0)
            steps = step_tm1 * (T.eq(test, 0) +
                                T.gt(test, 0) * self.increase +
                                diff * self.decrease)
            step = T.minimum(self.max_step, T.maximum(self.min_step, steps))
            grad = grad - diff * grad
            self.updates.append((param, c(param - T.sgn(grad) * step)))
            self.updates.append((grad_tm1, grad))
            self.updates.append((step_tm1, step))
        return self.updates
