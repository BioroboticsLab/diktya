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
from diktyo.func_api_helpers import get_layer, keras_copy, trainable, name_tensor


def _listify(x):
    if type(x) != list:
        return [x]
    else:
        return x


class GAN(AbstractModel):
    """
    Generative Adversarial Networks (GAN) are a unsupervised learning framework.
    It consists of a generator and a discriminator network. The generator
    recieves a noise vector as input and produces some fake data. The
    discriminator is trained to distinguish between fake data from the generator
    and real data. The generator is optimized to fool the discriminator.
    Please refere to `Goodwellow et. al <http://arxiv.org/abs/1406.2661>`_ for
    a detail introduction into GANs.

    Args:
        generator (Model): model of the generator. Must have one output and
            one input must be named `z`.
        discriminator (Model): model of the discriminator. Must have exaclty one input named
            `data`. For every sample, the output must be a scalar between 0 and 1.

    .. code:: python

        z = Input(shape=(20,), name='z')
        data = Input(shape=(1, 32, 32), name='real')

        n = 64
        fake = sequential([
            Dense(2*16*n, activation='relu'),
            Reshape(2*n, 4, 4),
        ])(z)

        realness = sequential([
            Convolution2D(n, 3, 3, border='same'),
            LeakyRelu(0.3),
            Flatten(),
            Dense(1),
        ])

        generator = Model(z, fake)
        generator.compile(Adam(lr=0.0002, beta_1=0.5), 'binary_crossentropy')

        discriminator = Model(data, realness)
        discriminator.compile(Adam(lr=0.0002, beta_1=0.5), 'binary_crossentropy')
        gan = GAN(generator, discriminator)

        gan.fit_generator(...)
    """
    def __init__(self, generator: Model, discriminator: Model):
        self.g = generator
        assert hasattr(self.g, 'optimizer'), "Did you forgot to call model.compile(...)?"

        self.d = discriminator
        assert hasattr(self.d, 'optimizer'), "Did you forgot to call model.compile(...)?"

        realness = name_tensor(self.d(self.g(self.g.inputs)), 'g_loss')
        self.fit_g = Model(self.g.inputs, realness)
        with trainable(self.d, False):
            self.fit_g.compile(keras_copy(self.g.optimizer), self.g.loss)
            self.fit_g._make_train_function()

        self.z = self.g.inputs[0]
        self.z_input_layer = get_layer(self.z)
        self.z_shape = self.z_input_layer.get_output_shape_at(0)

    def train_on_batch(self, inputs=None, generator_inputs=None, discriminator_inputs=None):
        """
        Runs a single weight update on a single batch of data.  Updates both
        generator and discriminator.

        Args:
            inputs (optional): Inputs for both the discriminator and the
                geneator. It can either be a numpy array, a list or dict.
                    * **numpy array**: ``real``
                    * **list**: ``[real]``, ``[real, z]``
                    * **dict**: ``{'real': real}``, ``{'real': real, 'z': z}``, ``{'real': real, 'z': z, 'additional_input', x}``
            generator_inputs (optional dict): This inputs will only be passed to
                the generator.
            discriminator_inputs (optional dict): This inputs will only be passed to
                the discriminator.

        Returns:
            A list of metrics. You can get the names of the metrics with
            :meth:`metrics_names`.

        """
        if type(inputs) is list:
            if len(inputs) == 1:
                inputs = {'real': inputs}
            if len(inputs) == 2:
                inputs = {'real': inputs[0], 'z': inputs[1]}
        elif type(inputs) is np.ndarray:
            inputs = {'real': inputs}

        if generator_inputs is None:
            generator_inputs = {k: v for k, v in inputs.items() if k != 'real'}
        if discriminator_inputs is None:
            discriminator_inputs = {k: v for k, v in inputs.items() if k != 'z'}
        if 'z' not in generator_inputs:
            generator_inputs['z'] = self.random_z(len(discriminator_inputs['real']))

        real = discriminator_inputs.pop('real')
        fake = self.g.predict(generator_inputs)
        g_loss = self.fit_g.train_on_batch(generator_inputs, np.ones((len(fake), 1)))

        discriminator_inputs['data'] = np.concatenate([fake, real])
        d_target = np.concatenate([
            np.zeros((len(fake), 1)),
            np.ones((len(real), 1))
        ])
        d_loss = self.d.train_on_batch(discriminator_inputs, d_target)
        return _listify(g_loss) + _listify(d_loss)

    @property
    def metrics_names(self):
        """
        Name of the metrics returned by :meth:`train_on_batch`.
        """
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
        """
        Fits the generator and discriminator on data generated by a Python
        generator. The generator is not run in parallel as in keras.

        Args:
            generator: the output of the generator must satisfy the
                train_on_batch method.
            nb_batches_per_epoch (int): run that many batches per epoch
            nb_epoch (int): run that many epochs
            batch_size (int): size of one batch
            verbose: verbosity mode
            callbacks: list of callbacks.

        """
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

    def generate(self, inputs=None, nb_samples=None):
        """
        Use the generator to generate data.

        Args:
           inputs: Dictionary of name to input arrays to the generator.
              Can include the random noise `z` or some conditional varialbes.
           nb_samples: Specifies how many samples will be generated, if `z` is
              not in the `inputs` dictionary.

        Returns:
            A numpy array with the generated data.
        """
        if inputs is None:
            inputs = {}
        if 'z' not in inputs:
            if nb_samples is None:
                raise Exception("`z` is not in the inputs dictonary and "
                                " nb_samples is also not given.")
            z_shape = self.z_shape
            if nb_samples:
                z_shape = (nb_samples, ) + z_shape[1:]
            assert None not in z_shape
            inputs['z'] = np.random.uniform(-1, 1, z_shape)

        return self.g.predict(inputs)

    def random_z(self, batch_size=32):
        """
        Samples `z` from uniform distribution between -1 and 1.
        The returned array is of shape ``(batch_size, ) + self.z_shape[1:]``
        """
        return np.random.uniform(-1, 1, (batch_size,) + self.z_shape[1:])

    def random_z_point(self):
        """
        Returns one random point in the z space.
        """
        shp = self.z_shape[1:]
        return np.random.uniform(-1, 1, shp)

    def interpolate(self, x, y, nb_steps=100):
        """
        Interpolates linear between two points in the z-space.

        Args:
            x: point in the z-space
            y: point in the z-space
            nb_steps: interpolate that many points

        Returns:
            The generated data from the interpolated points. The data
            corresponding to ``x`` and ``y`` are on the first and last position
            of the returned array.
        """
        assert x.shape == y.shape == self.z_shape[1:]
        z = np.zeros((nb_steps,) + x.shape)
        for i in range(nb_steps):
            z[i] = x + i / nb_steps * (y - x)
        return self.generate({'z': z})

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
