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

from collections import OrderedDict
from keras.objectives import binary_crossentropy
import keras.backend as K
from beras.layers.core import Split
import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from keras.models import Sequential
import keras.optimizers
from keras.optimizers import Optimizer
from keras.callbacks import Callback
from keras.engine.topology import Input, merge
from beras.models import AbstractModel
from beras.util import collect_layers


def flatten(listOfLists):
    return [el for list in listOfLists for el in list]


def gan_binary_crossentropy(d_out_given_fake_for_gen,
                            d_out_given_fake_for_dis,
                            d_out_given_real):
    d_loss_fake = binary_crossentropy(
        T.zeros_like(d_out_given_fake_for_dis),
        d_out_given_fake_for_dis).mean()
    d_loss_real = binary_crossentropy(
        T.ones_like(d_out_given_real),
        d_out_given_real).mean()
    d_loss = d_loss_real + d_loss_fake
    g_loss = binary_crossentropy(
        T.ones_like(d_out_given_fake_for_gen),
        d_out_given_fake_for_gen).mean()
    return g_loss, d_loss, d_loss_real, d_loss_fake


def gan_generator_kl(d_out_given_fake_for_gen,
                     d_out_given_fake_for_dis, d_out_given_real):
    """ see: http://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/  """
    d_loss_fake = binary_crossentropy(
        T.zeros_like(d_out_given_fake_for_dis),
        d_out_given_fake_for_dis).mean()
    d_loss_real = binary_crossentropy(
        T.ones_like(d_out_given_real),
        d_out_given_real).mean()
    d_loss = d_loss_real + d_loss_fake

    d = d_out_given_fake_for_gen
    e = 1e-7
    g_loss = - T.log(T.clip(d / (1 - d + e), e, 1 - e)).mean()
    return g_loss, d_loss, d_loss_real, d_loss_fake


def gan_generator_neg_log(d_out_given_fake_for_gen, d_out_given_fake_for_dis,
                          d_out_given_real):
    d_loss_fake = binary_crossentropy(
        T.zeros_like(d_out_given_fake_for_dis),
        d_out_given_fake_for_dis).mean()
    d_loss_real = binary_crossentropy(
        T.ones_like(d_out_given_real),
        d_out_given_real).mean()
    d_loss = d_loss_real + d_loss_fake

    d = d_out_given_fake_for_gen
    g_loss = - T.log(T.clip(d, 1e-7, 1 - 1e-7)).mean()
    return g_loss, d_loss, d_loss_real, d_loss_fake


def gan_linear_losses(d_out_given_fake_for_gen,
                      d_out_given_fake_for_dis, d_out_given_real):
    g_loss = -d_out_given_fake_for_gen.mean()
    d_loss_fake = d_out_given_fake_for_dis.mean()
    d_loss_real = d_out_given_real.mean()

    d_loss = d_loss_fake - d_loss_real + d_loss_fake**2 + d_loss_real**2
    return g_loss, d_loss, d_loss_real, d_loss_fake


def sequential_to_gan(generator: Sequential, discriminator: Sequential,
                      nb_real=32, nb_fake=96):
    def generator_fn(inputs):
        z, = inputs
        return generator(z)

    def discriminator_fn(inputs):
        fake, real = inputs
        dis_in = merge([fake, real], concat_axis=0, mode='concat')
        dis = discriminator(dis_in)
        dis_outputs = gan_outputs(dis, fake_for_gen=(0, nb_fake),
                                  fake_for_dis=(nb_fake - nb_real, nb_real),
                                  real=(nb_fake, nb_fake + nb_real))
        return dis_outputs
    return GAN(generator_fn, discriminator_fn,
               z_shape=generator.input_shape[1:],
               real_shape=discriminator.input_shape[1:])


def gan_outputs(discriminator,
                fake_for_gen=(0, 64),
                fake_for_dis=(0, 64),
                real=(64, 128)):
    return Split(*fake_for_gen)(discriminator), \
        Split(*fake_for_dis)(discriminator), \
        Split(*real)(discriminator)


class GAN(AbstractModel):
    dis_out_given_fake_for_gen_idx = 0
    dis_out_given_fake_for_dis_idx = 1
    dis_out_given_real_idx = 2
    real_idx = 1
    fake_idx = 0
    z_idx = 0

    class Regularizer(Callback):
        def __call__(self, g_loss, d_loss):
            return g_loss, d_loss

        def set_gan(self, gan):
            self.gan = gan

    class StopRegularizer(Callback):
        def __init__(self, high=1.3):
            self.high = K.variable(high)

        def set_gan(self, gan):
            self.gan = gan

        def __call__(self, g_loss, d_loss):
            d_loss = ifelse(T.and_(g_loss > self.high,
                                   d_loss < 1.5*self.high),
                            0.*d_loss,
                            d_loss)
            return g_loss, d_loss

    class L2Regularizer(Regularizer):
        def __init__(self, low=0.9, high=1.2, delta_value=5e-5, l2_init=0):
            self.l2_coef = theano.shared(
                np.cast['float32'](l2_init), name="l2_rate")
            self.low = low
            self.high = high
            self.delta = theano.shared(np.cast['float32'](delta_value),
                                       name="l2_rate")

        def _apply_l2_regulizer(self, params, loss):
            l2_loss = T.zeros_like(loss)
            for p in params:
                l2_loss += T.sum(p ** 2) * self.l2_coef
            return loss + l2_loss

        def __call__(self, g_loss, d_loss):
            small_delta = np.cast['float32'](1e-7)
            delta_l2 = ifelse(g_loss > self.high,
                              self.delta,
                              ifelse(g_loss < self.low,
                                     -self.delta,
                                     -small_delta))

            new_l2 = T.maximum(self.l2_coef + delta_l2, 0.)
            updates = [(self.l2_coef, new_l2)]
            self.gan.updates += updates
            params = flatten(
                [n.trainable_weights
                 for n in self.gan.get_discriminator_nodes().values()])
            d_loss = self._apply_l2_regulizer(params, d_loss)
            return g_loss, d_loss

    def __init__(self, generator: '(inputs) -> Container',
                 discriminator: '(inputs) -> Container',
                 z_shape, real_shape, gen_additional_inputs=[],
                 dis_additional_inputs=[],
                 non_trainable_layers=set()
                 ):
        self.generator_func = generator
        self.discriminator_func = discriminator

        self.z_shape = (None,) + z_shape
        self.real_shape = (None,) + real_shape
        self.z = Input(batch_shape=self.z_shape, name='z')
        self.real = Input(batch_shape=self.real_shape, name='real')

        self.gen_inputs = [self.z] + gen_additional_inputs
        self.fake = self.generator_func(self.gen_inputs)

        self.gen_layers = collect_layers(self.gen_inputs, [self.fake])
        self.dis_inputs = [self.fake, self.real] + dis_additional_inputs

        self._maybe_duplicate_inputs = self.gen_inputs + [self.real] \
            + dis_additional_inputs
        self.dis_outs = self.discriminator_func(self.dis_inputs)
        assert len(self.dis_outs) == 3, \
            "The discriminator must return 3 tensors"
        self.dis_layers = collect_layers(self.dis_inputs, self.dis_outs)
        self._gan_regularizers = []
        self.updates = []
        self.metrics = OrderedDict()
        self._is_built = False
        self.non_trainable_layers = set(non_trainable_layers)

    @property
    def inputs(self):
        inputs_set = set()
        inputs = []
        for input in self._maybe_duplicate_inputs:
            if input not in inputs_set:
                inputs_set.add(input)
                inputs.append(input)
        inputs.append(K.learning_phase())
        return inputs

    @property
    def input_order(self):
        return [i.name for i in self.inputs]

    @property
    def layers(self):
        return self.gen_layers + self.dis_layers

    def debug_dict(self):
        self._ensure_build()

        outs = OrderedDict([(l.name, l.output) for l in self.layers])

        outs.update([('{}_grad'.format(l.name), g) for l, g in
                     zip(self.gen_layers,
                         T.grad(self.g_loss,
                                [l.output for l in self.gen_layers],
                                disconnected_inputs='warn'))])

        outs.update([('{}_grad'.format(l.name), g) for l, g in
                     zip(self.dis_layers,
                         T.grad(self.d_loss,
                                [l.output for l in self.dis_layers],
                                disconnected_inputs='warn'))])
        return outs

    def _set_loss_metrics(self, g_loss, d_loss, d_loss_gen, d_loss_real):
        self.metrics['g_loss'] = g_loss
        self.metrics['d_loss'] = d_loss
        self.metrics['d_loss_gen'] = d_loss_gen
        self.metrics['d_loss_real'] = d_loss_real

    def _gather_list_attr(self, layers, attr):
        all_attrs = []
        for layer in layers:
            all_attrs += getattr(layer, attr, [])
        return all_attrs

    def _gather_dict_attr(self, layers, attr):
        all_attrs = {}
        for layer in layers:
            layer_dict = getattr(layer, attr, {})
            all_attrs = dict(list(all_attrs.items()) +
                             list(layer_dict.items()))
        return all_attrs

    def build(self, g_optimizer: Optimizer, d_optimizer: Optimizer,
              loss):
        assert not self._is_built

        def get_updates(optimizer, layers, loss):
            layers = set(layers) - set(self.non_trainable_layers)
            optimizer = keras.optimizers.get(optimizer)
            for r in self._gather_list_attr(layers, 'regularizers'):
                loss += r(loss)
            updates = optimizer.get_updates(
                self._gather_list_attr(layers, 'trainable_weights'),
                self._gather_dict_attr(layers, 'constraints'),
                loss)
            return updates + self._gather_list_attr(layers, 'updates')

        d_outs = self.dis_outs
        d_out_given_fake_gen = d_outs[self.dis_out_given_fake_for_gen_idx]
        d_out_given_fake_dis = d_outs[self.dis_out_given_fake_for_dis_idx]
        d_out_given_real = d_outs[self.dis_out_given_real_idx]

        losses = loss(
            d_out_given_fake_gen, d_out_given_fake_dis, d_out_given_real)

        g_loss, d_loss = losses[:2]
        for r in self._gan_regularizers:
            g_loss, d_loss = r(g_loss, d_loss)

        self._set_loss_metrics(g_loss, d_loss, *losses[2:])

        g_updates = get_updates(g_optimizer, self.gen_layers, g_loss)
        d_updates = get_updates(d_optimizer, self.dis_layers, d_loss)

        self.g_updates = g_updates
        self.d_updates = d_updates
        self.g_loss = g_loss
        self.d_loss = d_loss
        self._is_built = True

    def _ensure_build(self):
        if not self._is_built:
            self.build()

    def compile(self):
        assert self._is_built, \
            "Did you forget to call `build()`, before `compile`?"
        updates = self.g_updates + self.d_updates + self.updates
        self._train = K.function(self.inputs,
                                 list(self.metrics.values()),
                                 updates=updates)
        self.compile_generate()

    def compile_generate(self):
        assert self._is_built, \
            "Did you forget to call `build()`, before `compile_generate`?"
        self._generate = K.function(
            self.gen_inputs + [K.learning_phase()],
            self.fake)

    def compile_debug(self):
        assert self._is_built, \
            "Did you forget to call `build()`, before `compile_debug`?"
        self._debug = K.function(
            self.inputs, list(self.debug_dict().values()))

    def add_gan_regularizer(self, r):
        r.set_gan(self)
        self._gan_regularizers.append(r)

    def fit(self, data, nb_epoch=100, verbose=0, batch_size=128,
            callbacks=None):
        if type(batch_size) == int:
            batch_size = {name: batch_size for name in data.keys()}
        if callbacks is None:
            callbacks = []

        first_name = next(iter(data.keys()))
        nb_train_sample = len(data[first_name])
        nb_batches = nb_train_sample // batch_size[first_name]

        def train(model, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            ins = []
            for name in self.input_order:
                print(name)
                b = batch_size[name]
                e = b + batch_size[name]
                ins.append(data[name][b:e])
            outs = self._train(ins)
            for key, value in zip(self.metrics.keys(), outs):
                batch_logs[key] = value

        return self._fit(train, nb_train_sample=nb_train_sample,
                         nb_batches=nb_batches, nb_epoch=nb_epoch,
                         verbose=verbose, callbacks=callbacks, shuffle=False,
                         metrics=list(self.metrics.keys()))

    def fit_generator(self, generator, nb_batches_per_epoch,
                      nb_epoch, batch_size=128, verbose=1, callbacks=[]):
        if callbacks is None:
            callbacks = []

        def train(model, batch_index, batch_logs=None):
            ins = next(generator)
            ins['keras_learning_phase'] = 1
            inputs = [ins[name] for name in self.input_order]
            outs = self._train(inputs)
            for key, value in zip(self.metrics.keys(), outs):
                batch_logs[key] = value

        return self._fit(train,
                         nb_train_sample=batch_size*nb_batches_per_epoch,
                         nb_batches=nb_batches_per_epoch,
                         nb_epoch=nb_epoch, verbose=verbose,
                         callbacks=callbacks, shuffle=False,
                         metrics=list(self.metrics.keys()))

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

        inputs['keras_learning_phase'] = 0
        ins = [inputs[n] for n in self.input_order
               if n in inputs]
        assert len(ins) == len(self.gen_inputs) + 1
        return self._generate(ins)

    def debug(self, inputs={}, train=True):
        inputs['keras_learning_phase'] = int(train)
        ins = [inputs[n] for n in self.input_order]
        outs = self._debug(ins)
        return dict(zip(self.debug_dict().keys(), outs))

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
