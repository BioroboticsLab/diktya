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
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from keras.models import Sequential, standardize_X, Graph, model_from_json, \
    model_from_config
from keras import optimizers

from beras.models import AbstractModel, asgraph


class GAN(AbstractModel):
    z_name = "z"
    g_out_name = "g_out"
    d_input = 'd_input'

    class Regularizer(object):
        def get_losses(self, gan, g_loss, d_loss):
            updates = []
            return g_loss, d_loss, updates

    class L2Regularizer(Regularizer):
        def __init__(self, low=0.9, high=1.3):
            self.l2_coef = theano.shared(np.cast['float32'](0.), name="l2_rate")
            self.low = low
            self.high = high

        def _apply_l2_regulizer(self, params, loss):
            l2_loss = T.zeros_like(loss)
            for p in params:
                l2_loss += T.sum(p ** 2) * self.l2_coef
            return loss + l2_loss

        def get_losses(self, gan, g_loss, d_loss):
            delta = np.cast['float32'](1e-5)
            small_delta = np.cast['float32'](1e-7)
            delta_l2 = ifelse(g_loss > self.high,
                              delta,
                              ifelse(g_loss < self.low,
                                     -delta,
                                     -small_delta))

            new_l2 = T.maximum(self.l2_coef + delta_l2, 0.)
            updates = [(self.l2_coef, new_l2)]
            d_loss = self._apply_l2_regulizer(gan.D.params, d_loss)
            return g_loss, d_loss, updates

    def __init__(self, generator: Graph, discremenator: Graph, z_shape,
                 reconstruct_fn=None):
        """
        :param generator: Generator network. A keras Graph  model
        :param discremenator: Discriminator network. A keras Graph model
        :param z_shape: Shape of the random z vector.
        :return:
        """
        assert type(generator) == Graph, "generator must be of type Graph"
        assert type(discremenator) == Graph, \
            "discremenator must be of type Graph"
        self.z_shape = z_shape
        self.G = generator

        def filter_conds(name, inputs):
            return list(sorted(filter(lambda k: k != name, inputs.keys())))

        self.generator_conds = filter_conds(self.z_name, self.G.inputs)

        self.D = discremenator
        self.discriminator_conds = filter_conds(self.d_input, self.D.inputs)
        conditional_joined = set(self.discriminator_conds).union(
            set(self.generator_conds))
        self.conditionals = list(sorted(conditional_joined))
        if reconstruct_fn is None:
            self.reconstruct_fn = lambda g_outmap: g_outmap["output"]
        else:
            self.reconstruct_fn = reconstruct_fn

    @property
    def batch_size(self):
        return self.z_shape[0]

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
                        conditionals, labels, train=True):
        d_in = T.concatenate([fake, real], axis=0)
        self._set_input(self.D, [d_in] + conditionals,
                        [GAN.d_input] + labels)
        out = self.D.get_output(train)
        if type(out) is not dict:
            out = {"output": out}
        return out

    def losses(self, d_out, objective=binary_crossentropy):
        d_out_given_fake = d_out[:self.batch_size // 2]
        d_out_given_real = d_out[self.batch_size // 2:]
        d_loss_fake = objective(T.zeros_like(d_out_given_fake),
                                d_out_given_fake).mean()
        d_loss_real = objective(T.ones_like(d_out_given_real),
                                d_out_given_real).mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = objective(T.ones_like(d_out_given_fake),
                           d_out_given_fake).mean()
        return g_loss, d_loss, d_loss_real, d_loss_fake

    @staticmethod
    def _placeholder(x):
        # if is shared value
        if hasattr(x, 'get_value'):
            return K.placeholder(x.get_value().shape)
        else:
            return K.placeholder(ndim=x.ndim)

    def build_loss(self, z='random', objective=binary_crossentropy):
        objective = keras.objectives.get(objective)
        conditional_dict = OrderedDict()
        for c in self.conditionals:
            if c in self.G.inputs and c in self.D.inputs:
                c_g = self.G.inputs[c]
                c_d = self.D.inputs[c]
                assert c_g.input_shape == c_d.input_shape, \
                    "Got an input {} to the generator and discrimintor." \
                    " But they have different shape." \
                    " Got {} for the generator and {} for the discriminator." \
                    .format(c, c_g.shape, c_d.shape)
                shape = c_g.shape
            if c in self.G.inputs:
                shape = self.G.inputs[c].input_shape
            else:
                assert c in self.D.inputs
                shape = self.D.inputs[c].input_shape
            conditional_dict[c] = K.placeholder(shape=shape, name=c)

        conditionals = list(conditional_dict.values())
        gen_conditionals = [conditional_dict[n] for n in self.generator_conds]
        dis_conditionals = [conditional_dict[n]
                            for n in self.discriminator_conds]

        z = self._get_z(z)
        g_outmap = self._get_gen_output(z, gen_conditionals,
                                        self.generator_conds)
        g_out = g_outmap["output"]
        fake = self.reconstruct_fn(g_outmap)
        real = K.placeholder(ndim=fake.ndim, name="real")
        d_outmap = self._get_dis_output(fake, real, dis_conditionals,
                                        self.discriminator_conds)
        d_out = d_outmap["output"]
        g_loss, d_loss, d_loss_real, d_loss_gen = self.losses(d_out, objective)

        placeholder_z = self._placeholder(z)
        replace_z = [(z, placeholder_z)]
        return DotMap(locals())

    @staticmethod
    def get_regulizer(self, regulizer=None) -> Regularizer:
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
        v.g_loss, v.d_loss, v.reg_updates = gan_regulizer.get_losses(
                self, v.g_loss, v.d_loss)

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
                  z_type='random'):
        v = self.build_loss(z_type)
        self.build_regulizer(v, gan_regulizer)
        self.build_opt_d(optimizer_d, v)
        self.build_opt_g(optimizer_g, v)
        return v

    def _compile_generate(self, v, mode=None):
        add_inputs = v.gen_conditionals
        self._generate = theano.function(
                [v.placeholder_z] + add_inputs,
                [v.fake],
                allow_input_downcast=True,
                mode=mode, givens=v.replace_z)

    def _compile_debug(self, v, mode=None):
        self._debug = theano.function(
                [v.real, v.placeholder_z] + v.conditionals,
                [v.g_out, v.fake, v.real, v.d_loss, v.d_loss_real,
                 v.d_loss_gen,
                 v.g_loss],
                allow_input_downcast=True, mode=mode, givens=v.replace_z)

    def compile(self, optimizer_g, optimizer_d, gan_regulizer=None, mode=None):
        import pytest
        v = self.build_opt(optimizer_g, optimizer_d, gan_regulizer)
        # pytest.set_trace()
        self._train = theano.function(
                [v.real] + v.conditionals,
                [v.d_loss, v.d_loss_real, v.d_loss_gen, v.g_loss],
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
                       callbacks=None, verbose=0, conditionals=[]):
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

            ins = [expected_image] + conditionals
            outs = self._optimize_image_fn(*ins)
            for key, value in zip(labels, outs):
                batch_logs[key] = value

        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(optimize, self.batch_size * nb_iterations,
                  batch_size=self.batch_size, nb_epoch=1, verbose=verbose,
                  callbacks=callbacks, shuffle=False, metrics=labels)
        return self.generate(z.get_value()), z.get_value()

    def fit(self, X, conditional_inputs=None, nb_epoch=100, verbose=0,
            callbacks=None, shuffle=True):
        if callbacks is None:
            callbacks = []
        conditionals = []
        if conditional_inputs is not None:
            for _, array in sorted(conditional_inputs.items()):
                conditionals.append(array)
        labels = ['d_loss', 'd_real', 'd_gen', 'g_loss']

        def train(model, batch_ids, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}
            ins = [X[batch_ids]]
            for c in conditionals:
                print(batch_ids.shape)
                ins.append(c[batch_ids])
            outs = self._train(*ins)
            for key, value in zip(labels, outs):
                batch_logs[key] = value

        assert self.batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, len(X), batch_size=self.batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle,
                  metrics=labels)

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

    def generate(self, z=None, conditionals=[]):
        if z is None:
            z = self._uniform_z()
        ins = [z] + list(conditionals)
        return self._generate(*ins)[0]

    def debug(self, X, z=None, *conditionals):
        if z is None:
            z = self._uniform_z()
        labels = ['g_out', 'fake', 'real', 'd_loss', 'd_real',
                  'd_gen', 'g_loss']
        ins = [X, z] + list(conditionals)
        outs = self._debug(*ins)
        return DotMap(dict(zip(labels, outs)))

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

    @property
    def batch_size(self):
        return self.z_shape[0]

    def g_output_shape(self):
        return (self.batch_size,) + self.G.output_shape[1:]
