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

import json
import os
from collections import OrderedDict
import keras
from dotmap import DotMap
from keras.objectives import binary_crossentropy
import keras.backend as K
from keras.backend.common import cast_to_floatx

import theano
from theano.tensor import TensorType
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from keras.models import Graph, model_from_config
from keras import optimizers
from keras.callbacks import Callback
from beras.models import AbstractModel


class GAN(AbstractModel):
    z_name = "z"
    g_out_name = "g_out"
    d_input = 'd_input'

    class Regularizer(Callback):
        def get_losses(self, gan, g_loss, d_loss):
            updates = []
            metrics = {}
            return g_loss, d_loss, updates, metrics

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

        def get_losses(self, gan, g_loss, d_loss):
            small_delta = np.cast['float32'](1e-7)
            delta_l2 = ifelse(g_loss > self.high,
                              self.delta,
                              ifelse(g_loss < self.low,
                                     -self.delta,
                                     -small_delta))

            new_l2 = T.maximum(self.l2_coef + delta_l2, 0.)
            updates = [(self.l2_coef, new_l2)]
            d_loss = self._apply_l2_regulizer(gan.D.params, d_loss)
            return g_loss, d_loss, updates, {'l2_reg': self.l2_coef.mean()}

    def __init__(self, generator: Graph, discriminator: Graph, z_shape,
                 batch_size=128, reconstruct_fn=None):
        """
        :param generator: Generator network. A keras Graph  model
        :param discriminator: Discriminator network. A keras Graph model
        :param z_shape: Shape of the random z vector.
        :return:
        """
        assert type(generator) == Graph, "generator must be of type Graph"
        assert type(discriminator) == Graph, \
            "discriminator must be of type Graph"
        self.z_shape = z_shape
        self.batch_size = batch_size
        self.G = generator

        def filter_conds(name, inputs):
            return list(sorted(filter(lambda k: k != name, inputs.keys())))

        self.generator_conds = filter_conds(self.z_name, self.G.inputs)

        self.D = discriminator
        self.discriminator_conds = filter_conds(self.d_input, self.D.inputs)
        conditional_joined = set(self.discriminator_conds).union(
            set(self.generator_conds))
        self.conditionals = list(sorted(conditional_joined))
        if reconstruct_fn is None:
            self.reconstruct_fn = lambda g_outmap: g_outmap["output"]
        else:
            self.reconstruct_fn = reconstruct_fn

    @staticmethod
    def _set_input(model, inputs, labels):
        assert len(inputs) == len(labels)
        assert len(model.inputs) == len(inputs), \
            "inputs do not match. Got {} and {}"\
            .format(list(model.inputs.keys()), labels)
        if len(inputs) == 1:
            only_input = next(iter(model.inputs.values()))
            only_input.input = inputs[0]
        else:
            for label, input in zip(labels, inputs):
                model.inputs[label].input = input

    def _random_z(self):
        return K.random_uniform(self.z_shape, -1, 1)

    def _shared_z(self):
        return theano.shared(cast_to_floatx(np.zeros(self.z_shape)))

    def _get_z(self, z_type):
        if hasattr(z_type, 'ndim'):
            return z_type
        elif z_type == 'random':
            return self._random_z()
        elif z_type == 'shared':
            return self._shared_z()
        else:
            raise ValueError("must be either `random` or `shared`, "
                             "got: {}".format(z_type))

    def _get_gen_output(self, z, conditionals, labels, train=True):
        self._set_input(self.G, [z] + conditionals, [self.z_name] + labels)
        out = self.G.get_output(train)
        if type(out) is not dict:
            out = {"output": out}
        return out

    def _get_dis_output(self, fake, real,
                        conditionals, labels, d_image_views=1, train=True):
        fake_shape = fake.shape

        d_in = T.concatenate([fake, real], axis=0)
        d_in = d_in.reshape((-1, fake_shape[1], fake_shape[2],
                             fake_shape[3] * d_image_views))
        self._set_input(self.D, [d_in] + conditionals,
                        [GAN.d_input] + labels)
        out = self.D.get_output(train)
        if type(out) is not dict:
            out = {"output": out}
        return out

    def d_out_given_fake_for_gen(self, d_out, d_image_views=1):
        return d_out[:self.batch_size // (d_image_views)]

    def d_out_given_fake_for_dis(self, d_out, d_image_views=1):
        bs = self.batch_size // d_image_views
        return d_out[bs:bs+bs//2]

    def d_out_given_real(self, d_out, d_image_views=1):
        bs = self.batch_size // d_image_views
        return d_out[bs + bs//2:]

    def linear_losses(self, d_out, objective=None, d_image_views=1):
        g_loss = -self.d_out_given_fake_for_gen(d_out, d_image_views).mean()
        d_out_given_fake = self.d_out_given_fake_for_dis(d_out, d_image_views)
        d_out_given_real = self.d_out_given_real(d_out, d_image_views)
        d_loss_fake = d_out_given_fake.mean()
        d_loss_fake = d_loss_fake + d_loss_fake**2
        d_loss_real = -d_out_given_real.mean()
        d_loss_real = d_loss_real + d_loss_real**2
        d_loss = d_loss_real + d_loss_fake
        return g_loss, d_loss, d_loss_real, d_loss_fake

    def losses(self, d_out, objective=binary_crossentropy, d_image_views=1):
        d_out_given_fake_gan = self.d_out_given_fake_for_gen(
            d_out, d_image_views)
        d_out_given_fake_dis = self.d_out_given_fake_for_dis(
            d_out, d_image_views)
        d_out_given_real = self.d_out_given_real(d_out, d_image_views)
        d_loss_fake = objective(T.zeros_like(d_out_given_fake_dis),
                                d_out_given_fake_dis).mean()
        d_loss_real = objective(T.ones_like(d_out_given_real),
                                d_out_given_real).mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = objective(T.ones_like(d_out_given_fake_gan),
                           d_out_given_fake_gan).mean()
        return g_loss, d_loss, d_loss_real, d_loss_fake

    @staticmethod
    def _placeholder(x, name=None):
        # if is shared value
        if hasattr(x, 'get_value'):
            return K.placeholder(x.get_value().shape, name=name)
        else:
            return TensorType(x.dtype, x.broadcastable)()

    def conditionals_dict(self):
        conditional_dict = OrderedDict()
        for c in self.conditionals:
            if c in self.G.inputs and c in self.D.inputs:
                c_g = self.G.inputs[c]
                c_d = self.D.inputs[c]
                assert c_g.input_shape == c_d.input_shape, \
                    "Got an input {} to the generator and discrimintor." \
                    " But they have different shape." \
                    " Got {} for the generator and {} for the discriminator." \
                    .format(c, c_g.input_shape, c_d.input_shape)
                shape = c_g.shape
            if c in self.G.inputs:
                shape = self.G.inputs[c].input_shape
            else:
                assert c in self.D.inputs
                shape = self.D.inputs[c].input_shape
            conditional_dict[c] = K.placeholder(shape=shape, name=c)
        return conditional_dict

    def build_loss(self, z='random', objective=binary_crossentropy,
                   d_loss_grad_weight=0, d_image_views=1,
                   use_linear_losses=False):
        objective = keras.objectives.get(objective)
        conditionals_dict = self.conditionals_dict()
        conditionals = list(conditionals_dict.values())
        gen_conditionals = [conditionals_dict[n] for n in self.generator_conds]
        dis_conditionals = [conditionals_dict[n]
                            for n in self.discriminator_conds]

        z = self._get_z(z)
        g_outmap = self._get_gen_output(z, gen_conditionals,
                                        self.generator_conds)
        g_out = g_outmap["output"]
        fake = self.reconstruct_fn(g_outmap)
        real = K.placeholder(ndim=fake.ndim, name="real")
        d_outmap = self._get_dis_output(
            fake, real, dis_conditionals,
            self.discriminator_conds, d_image_views)
        d_out = d_outmap["output"]

        loss_fn = self.losses
        if use_linear_losses:
            loss_fn = self.linear_losses
        g_loss, d_loss, d_loss_real, d_loss_gen = loss_fn(
            d_out, objective, d_image_views)

        if d_loss_grad_weight != 0:
            fake_grad = T.grad(g_loss, fake)
            d_loss_grad = fake_grad.norm(2)
            d_loss_grad *= d_loss_grad_weight
            d_loss += d_loss_grad

        metrics = {'d_loss': d_loss, 'd_real': d_loss_real,
                   'd_gen': d_loss_gen, 'g_loss': g_loss}
        placeholder_z = self._placeholder(z, name="z_placeholder")
        replace_z = [(z, placeholder_z)]
        return DotMap(locals())

    @staticmethod
    def get_regulizer(regulizer=None):
        if regulizer is None:
            return GAN.Regularizer()
        elif regulizer == 'l2':
            return GAN.L2Regularizer()
        elif issubclass(type(regulizer), GAN.Regularizer):
            return regulizer
        elif type(regulizer) == GAN.Regularizer:
            return regulizer
        else:
            raise ValueError("Cannot get regulizer for value `{}`"
                             .format(regulizer))

    def build_regulizer(self, v_loss_map, gan_regulizer=None):
        v = v_loss_map
        gan_regulizer = self.get_regulizer(gan_regulizer)
        v.g_loss, v.d_loss, v.reg_updates, metrics = gan_regulizer.get_losses(
                self, v.g_loss, v.d_loss)
        v.metrics.d_loss = v.d_loss
        v.metrics.update(metrics)

    def build_opt_g(self, optimizer, v_loss_map):
        v = v_loss_map
        optimizer = optimizers.get(optimizer)
        v.g_updates = optimizer.get_updates(
                self.G.params, self.G.constraints, v.g_loss)

    def build_opt_d(self, optimizer, v_loss_map):
        v = v_loss_map
        optimizer = optimizers.get(optimizer)
        v.d_updates = optimizer.get_updates(
                self.D.params, self.D.constraints, v.d_loss)

    def build_opt(self, optimizer_g, optimizer_d, gan_regulizer=None,
                  z_type='random', d_image_views=1, use_linear_losses=False):
        v = self.build_loss(z_type, d_image_views=d_image_views,
                            use_linear_losses=use_linear_losses)
        print(gan_regulizer)
        self.build_regulizer(v, gan_regulizer)
        self.build_opt_d(optimizer_d, v)
        self.build_opt_g(optimizer_g, v)
        return v

    def _compile_generate(self, v, mode=None):
        self._generate = theano.function(
                [v.placeholder_z] + v.gen_conditionals,
                [v.fake],
                allow_input_downcast=True,
                mode=mode, givens=v.replace_z)

    def debug_output(self, v):
        v_labels = ['g_out', 'fake', 'real', 'd_loss', 'd_loss_real',
                    'd_loss_gen', 'g_loss']
        if 'd_loss_grad' in v:
            v_labels.extend(['d_loss_grad', 'fake_grad'])

        d = [(label, v[label]) for label in v_labels]
        d.extend(list(v.conditionals_dict.items()))
        return OrderedDict(d)

    def _compile_debug(self, v, mode=None):
        debug_dict = self.debug_output(v)
        self._debug_labels = list(debug_dict.keys())
        print(list(debug_dict.items()))
        self._debug = theano.function(
            [v.real, v.placeholder_z] + v.conditionals,
            list(debug_dict.values()),
            allow_input_downcast=True, mode=mode, givens=v.replace_z)

    def compile(self, optimizer_g, optimizer_d, gan_regulizer=None,
                d_image_views=1, use_linear_losses=False, mode=None):
        v = self.build_opt(optimizer_g, optimizer_d, gan_regulizer,
                           d_image_views=d_image_views,
                           use_linear_losses=use_linear_losses)
        self.train_labels = list(sorted(v.metrics.keys()))
        self._train = theano.function(
                [v.real] + v.conditionals,
                [m for _, m in sorted(v.metrics.items())],
                updates=v.d_updates + v.g_updates + v.reg_updates,
                allow_input_downcast=True, mode=mode)

        self._compile_generate(v, mode)
        self._compile_debug(v, mode)

    def compile_optimize_image(self, optimizer, image_loss_fn, ndim_expected=4,
                               mode=None):
        v = self.build_loss(z='shared')
        optimizer = optimizers.get(optimizer)
        self.build_image_loss_vars = v
        out_expected = K.placeholder(ndim=ndim_expected)
        v.image_loss = image_loss_fn(out_expected, v.g_out)
        v.image_updates = optimizer.get_updates([v.z], self.D.constraints,
                                                v.image_loss)

        self._optimize_image_fn = theano.function(
                [out_expected] + v.gen_conditionals, [v.image_loss],
                updates=v.image_updates, allow_input_downcast=True, mode=mode)

        self._compile_generate(v, mode)
        self._compile_debug(v, mode)

    def _uniform_z(self):
        return cast_to_floatx(np.random.uniform(-1, 1, self.z_shape))

    def optimize_image(self, expected_image, nb_iterations, z_start=None,
                       callbacks=None, verbose=0, conditionals=None):
        if z_start is None:
            z_start = self._uniform_z()

        z = self.build_image_loss_vars.z
        z.set_value(cast_to_floatx(z_start))
        if callbacks is None:
            callbacks = []
        labels = ['loss']

        def optimize(model, batch_ids, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            ins = [expected_image] + self._conditionals_to_list(conditionals)
            outs = self._optimize_image_fn(*ins)
            for key, value in zip(labels, outs):
                batch_logs[key] = value

        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(optimize, self.batch_size * nb_iterations,
                  batch_size=self.batch_size, nb_epoch=1, verbose=verbose,
                  callbacks=callbacks, shuffle=False, metrics=labels)
        return self.generate(z.get_value()), z.get_value()

    def _conditionals_to_list(self, conds):
        if conds is None:
            return []
        else:
            l = []
            for name, array in sorted(conds.items()):
                if name not in self.conditionals:
                    raise ValueError(
                        "Expected conditionals to be one of {}, got {}"
                        .format(",".join(self.conditionals), name))
                l.append(array)
            return l

    def fit(self, X, conditional_inputs=None, nb_epoch=100, verbose=0,
            callbacks=None, shuffle=True):
        if callbacks is None:
            callbacks = []
        conditionals = self._conditionals_to_list(conditional_inputs)

        def train(model, batch_ids, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}
            ins = [X[batch_ids]]
            for c in conditionals:
                ins.append(c[batch_ids])
            outs = self._train(*ins)
            for key, value in zip(self.train_labels, outs):
                batch_logs[key] = value

        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, len(X), batch_size=self.batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle,
                  metrics=self.train_labels)

    def fit_generator(self, generator, samples_per_epoch,
                      nb_epoch, verbose=1, callbacks=[]):
        if callbacks is None:
            callbacks = []

        def train(model, batch_ids, batch_index, batch_logs=None):
            ins = next(generator)
            X = ins['real']
            del ins['real']
            conditionals = self._conditionals_to_list(ins)
            inputs = [X] + conditionals
            outs = self._train(*inputs)
            for key, value in zip(self.train_labels, outs):
                batch_logs[key] = value

        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, samples_per_epoch, batch_size=self.batch_size,
                  nb_epoch=nb_epoch, verbose=verbose, callbacks=callbacks,
                  shuffle=False, metrics=self.train_labels)

    def print_svg(self):
        theano.printing.pydotprint(self._g_train, outfile="train_g.png")
        theano.printing.pydotprint(self._d_train, outfile="train_d.png")

    def load_weights(self, fname):
        self.G.load_weights(fname.format("generator"))
        self.D.load_weights(fname.format("detector"))

    def save_weights(self, fname, overwrite=False):
        self.G.save_weights(fname.format("generator"), overwrite)
        self.D.save_weights(fname.format("detector"), overwrite)

    @staticmethod
    def _weight_fname_tmpl(directory):
        return os.path.join(directory, "{}.hdf5")

    @staticmethod
    def load(directory):
        with open(directory + "/gan.json") as f:
            config = json.load(f)
            gan = GAN.load_from_config(config)
        gan.load_weights(GAN._weight_fname_tmpl(directory))
        return gan

    @staticmethod
    def load_from_config(config):
        G = model_from_config(config['G'])
        D = model_from_config(config['D'])
        return GAN(G, D,
                   config['z_shape'])

    def get_config(self, verbose=0):
        g_config = self.G.get_config(verbose)
        d_config = self.D.get_config(verbose)
        return {
            'G': g_config,
            'D': d_config,
            'z_shape': self.z_shape,
        }

    def save(self, directory, overwrite=False):
        os.makedirs(directory, exist_ok=True)
        with open(directory + "/gan.json", "w") as f:
            json.dump(self.get_config(), f)
        self.save_weights(self._weight_fname_tmpl(directory), overwrite)

    def generate(self, z=None, conditionals=None):
        if z is None:
            z = self._uniform_z()
        ins = [z] + self._conditionals_to_list(conditionals)
        return self._generate(*ins)[0]

    def debug(self, X, z=None, conditionals={}):
        if z is None:
            z = self._uniform_z()
        ins = [X, z] + self._conditionals_to_list(conditionals)
        outs = self._debug(*ins)
        return DotMap(dict(zip(self._debug_labels, outs)))

    def interpolate(self, x, y):
        z = np.zeros(self.z_shape)
        n = len(z)
        for i in range(n):
            z[i] = x + i / n * (y - x)
        real = np.zeros(self.g_output_shape())
        outs = self.debug(real, z)
        return outs.fake

    def random_z_point(self):
        """returns a random point in the z space"""
        shp = self.z_shape[1:]
        return np.random.uniform(-1, 1, shp)

    def neighborhood(self, z_point=None, std=0.25):
        """samples the neighborhood of a z_point by adding gaussian noise
         to it. You can control the standard derivation of the noise with std."""
        shp = self.z_shape[1:]
        if z_point is None:
            z_point = np.random.uniform(-1, 1, shp)
        n = self.z_shape[0]
        z = np.zeros(self.z_shape)
        for i in range(n):
            offset = np.random.normal(0, std, shp)
            z[i] = np.clip(z_point + offset, -1, 1)

        real = np.zeros(self.g_output_shape())
        outs = self.debug(real, z)
        return outs.fake

    def train_batch(self, X, ZD, ZG, k=1):
        for i in range(k):
            self._d_train([X, ZD])
        self._d_train([ZG])

    def g_output_shape(self):
        return (self.batch_size,) + self.G.output_shape[1:]
