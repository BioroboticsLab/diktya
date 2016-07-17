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

import numpy as np
from keras.engine.training import Model
from diktyo.models import AbstractModel
from diktyo.util import get_layer, keras_copy, trainable, name_tensor


def _listify(x):
    if type(x) != list:
        return [x]
    else:
        return x


class GAN(AbstractModel):
    real_idx = 1
    fake_idx = 0
    z_idx = 0

    def __init__(self, generator: Model, discriminator: Model):
        self.g = generator
        assert hasattr(self.g, 'optimizer'), "Did you forgot to call model.compile(...)?"

        self.d = discriminator
        assert hasattr(self.d, 'optimizer'), "Did you forgot to call model.compile(...)?"

        self.fit_g = Model(self.g.inputs, name_tensor(self.d(self.g(self.g.inputs)), 'g_loss'))
        with trainable(self.d, False):
            self.fit_g.compile(keras_copy(self.g.optimizer), self.g.loss)

        self.z = self.g.inputs[0]
        self.z_input_layer = get_layer(self.z)
        self.z_shape = self.z_input_layer.get_output_shape_at(0)

    def train_on_batch(self, inputs=None, g_inputs=None, d_inputs=None):
        if g_inputs is None:
            g_inputs = {k: v for k, v in inputs.items() if k != 'real'}
        if d_inputs is None:
            d_inputs = {k: v for k, v in inputs.items() if k != 'z'}

        real = d_inputs.pop('real')
        fake = self.g.predict(g_inputs)

        with trainable(self.d, False):
            g_loss = self.fit_g.train_on_batch(g_inputs, np.ones((len(fake), 1)))

        d_inputs['data'] = np.concatenate([fake, real])

        d_target = np.concatenate([
            np.zeros((len(fake), 1)),
            np.ones((len(real), 1))
        ])
        d_loss = self.d.train_on_batch(d_inputs, d_target)
        return _listify(g_loss) + _listify(d_loss)

    @property
    def metrics_names(self):
        def replace(l, X, Y):
            for i, v in enumerate(l):
                if v == X:
                    l.pop(i)
                    l.insert(i, Y)
            return l

        g_metrics = replace(_listify(self.g.metrics_names), 'loss', 'g_loss')
        d_metrics = replace(_listify(self.d.metrics_names), 'loss', 'd_loss')
        return g_metrics + d_metrics

    def fit_generator(self, generator, nb_batches_per_epoch,
                      nb_epoch, batch_size=128, verbose=1, callbacks=[]):
        if callbacks is None:
            callbacks = []

        def train(model, batch_index, batch_logs=None):
            ins = next(generator)
            outs = self.train_on_batch(ins)
            for key, value in zip(self.metrics_names, outs):
                batch_logs[key] = value

        return self._fit(train,
                         nb_train_sample=batch_size*nb_batches_per_epoch,
                         nb_batches=nb_batches_per_epoch,
                         nb_epoch=nb_epoch, verbose=verbose,
                         callbacks=callbacks, shuffle=False,
                         metrics=self.metrics_names)

    def generate(self, inputs=None, z_shape=None, nb_samples=None):
        if inputs is None:
            inputs = {}
        if 'z' not in inputs:
            if z_shape is None:
                z_shape = self.z_shape
                if nb_samples:
                    z_shape = (nb_samples, ) + z_shape[1:]
            assert None not in z_shape
            inputs['z'] = np.random.uniform(-1, 1, z_shape)

        return self.g.predict(inputs)

    def interpolate(self, x, y, nb_steps=100):
        assert x.shape == y.shape == self.z_shape[1:]
        z = np.zeros((nb_steps,) + x.shape)
        for i in range(nb_steps):
            z[i] = x + i / nb_steps * (y - x)
        return self.generate({'z': z})

    def random_z_point(self):
        """returns a random point in the z space"""
        shp = self.z_shape[1:]
        return np.random.uniform(-1, 1, shp)

    def neighborhood(self, z_point=None, std=0.25, n=128):
        """samples the neighborhood of a z_point by adding gaussian noise
         to it. You can control the standard derivation of the noise with std.
          """
        shp = self.z_shape[1:]
        if z_point is None:
            z_point = np.random.uniform(-1, 1, shp)
        z = np.zeros((n,) + shp)
        for i in range(n):
            offset = np.random.normal(0, std, shp)
            z[i] = np.clip(z_point + offset, -1, 1)

        return self.generate({'z': z})

    def save_weights(self, fname, overwrite=False):
        self.g.save_weights(fname.format("generator"), overwrite)
        self.d.save_weights(fname.format("discriminator"), overwrite)

    def load_weights(self, fname):
        self.g.load_weights(fname.format("generator"))
        self.d.load_weights(fname.format("discriminator"))
