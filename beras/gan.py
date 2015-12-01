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
import inspect
import json
import os

from dotmap import DotMap
from keras.objectives import mse, binary_crossentropy
from keras.utils.theano_utils import ndim_tensor, floatX
import theano
import theano.tensor.shared_randomstreams as T_random
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from keras.models import Sequential, standardize_X, Graph, model_from_json, \
    model_from_config
from keras import optimizers
from theano.tensor.type import TensorType

from beras.models import AbstractModel

_rs = T_random.RandomStreams(1334)


class GAN(AbstractModel):
    ndim = 4

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
            delta = np.cast['float32'](0.00001)
            delta_l2 = ifelse(g_loss > self.high,
                            delta,
                            ifelse(g_loss < self.low,
                                   -delta,
                                   np.cast['float32'](0.)))

            new_l2 = T.maximum(self.l2_coef + delta_l2, 0.)
            updates = [(self.l2_coef, new_l2)]
            d_loss = self._apply_l2_regulizer(gan.D.params, d_loss)
            return g_loss, d_loss, updates

    def __init__(self, generator, discremenator,
                 z_shape, num_gen_conditional=0, num_dis_conditional=0,
                 num_both_conditional=0):
        self.num_dis_conditional = num_dis_conditional
        self.G = generator
        self.D = discremenator
        self.num_gen_conditional = num_gen_conditional
        self.num_both_conditional = num_both_conditional
        self.rs = T_random.RandomStreams(1334)

        self.z_shape = z_shape

    @staticmethod
    def _set_input(model, inputs, labels):
        if type(model) == Sequential:
            if type(inputs) == list or type(inputs) == tuple:
                if len(inputs) != 1:
                    input = T.concatenate(inputs, axis=1)
                else:
                    input = inputs[0]
            else:
                input = inputs
            model.layers[0].input = input
        elif type(model) == Graph:
            for label, input in zip(labels, inputs):
                model.inputs[label].input = input
        else:
            ValueError("model must be either Graph of Sequential")

    @staticmethod
    def _get_output(model, train=True):
        out = model.get_output(train)
        if type(out) == dict:
            return out["output"]
        else:
            return out

    def _random_z(self):
        return self.rs.uniform(self.z_shape, -1, 1)

    def _shared_z(self):
        return self.rs.uniform(self.z_shape, -1, 1)

    def _get_gen_output(self, gen_conditional, train=True):
        gen_conditional = standardize_X(gen_conditional)
        self.z = self._random_z()
        self.z_label = "z"
        self._set_input(self.G, [self.z] + gen_conditional,
                        [self.z_label] +
                        ["cond_{}".format(i)
                         for i in range(len(gen_conditional))])
        return self._get_output(self.G, train)

    def _get_dis_output(self, fake, real, dis_conditional, train=True):
        dis_conditional = standardize_X(dis_conditional)
        d_in = T.concatenate([fake, real], axis=0)
        self._set_input(self.D, [d_in] + dis_conditional,
                        ["input"] +
                        ["cond_{}".format(i)
                         for i in range(len(dis_conditional))])
        return self._get_output(self.D, train)

    def losses(self, real, gen_conditional=[], dis_conditional=[],
               both_conditional=[], gen_out=None,
               objective=binary_crossentropy):
        if gen_out is None:
            gen_out = self._get_gen_output(gen_conditional + both_conditional)

        self.g_out = gen_out
        self.d_out = self._get_dis_output(gen_out, real,
                                          dis_conditional + both_conditional)

        batch_size = self.d_out.shape[0]
        d_out_given_fake = self.d_out[:batch_size // 2]
        d_out_given_real = self.d_out[batch_size // 2:]
        d_loss_fake = objective(T.zeros_like(d_out_given_fake),
                                d_out_given_fake).mean()
        d_loss_real = objective(T.ones_like(d_out_given_real),
                                d_out_given_real).mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = objective(T.ones_like(d_out_given_fake),
                           d_out_given_fake).mean()
        return g_loss, d_loss, d_loss_real, d_loss_fake

    def build(self, optimizer_g=None, optimizer_d=None, gan_regulizer=None, mode=None, ndim_gen_out=4):
        if gan_regulizer is None:
            gan_regulizer = GAN.Regularizer()
        elif gan_regulizer == 'l2':
            gan_regulizer = GAN.L2Regularizer()

        x_real = ndim_tensor(ndim_gen_out)
        gen_conditional = [T.tensor4("gen_conditional_{}".format(i))
                           for i in range(self.num_gen_conditional)]
        both_conditional = [T.tensor4("conditional_{}".format(i))
                            for i in range(self.num_both_conditional)]
        dis_conditional = [T.tensor4("dis_conditional_{}".format(i))
                           for i in range(self.num_dis_conditional)]

        g_loss, d_loss, d_loss_real, d_loss_gen = \
            self.losses(x_real, gen_conditional, dis_conditional,
                        both_conditional)
        z = self.z
        g_out = self.g_out
        g_loss, d_loss, reg_updates = gan_regulizer.get_losses(self, g_loss, d_loss)

        if optimizer_d and optimizer_g:
            optimizer_g = optimizers.get(optimizer_g)
            optimizer_d = optimizers.get(optimizer_d)
            g_updates = optimizer_g.get_updates(
                self.G.params, self.G.constraints, g_loss)
            d_updates = optimizer_d.get_updates(
                self.D.params, self.D.constraints, d_loss)

        self.vars = DotMap(locals())

    def compile(self, optimizer_g, optimizer_d, gan_regulizer=None, mode=None, ndim_gen_out=4):
        if not hasattr(self, 'vars'):
            self.build(optimizer_g, optimizer_d, gan_regulizer, mode, ndim_gen_out)
        v = self.vars
        self._train = theano.function(
            [v.x_real] + v.gen_conditional + v.dis_conditional + v.both_conditional,
            [v.d_loss, v.d_loss_real, v.d_loss_gen, v.g_loss],
            updates=v.d_updates + v.g_updates + v.reg_updates,
            allow_input_downcast=True, mode=mode)
        self._generate = theano.function(
            v.gen_conditional + v.dis_conditional + v.both_conditional,
            [self.g_out],
            allow_input_downcast=True,
            mode=mode)

        v.placeholder_z = TensorType(self.z.dtype, self.z.broadcastable)()
        v.replace_z = [(self.z, v.placeholder_z)]
        self._debug = theano.function(
            [v.x_real, v.placeholder_z] + v.gen_conditional + v.dis_conditional + v.both_conditional,
            [v.g_out, v.x_real, v.d_loss, v.d_loss_real, v.d_loss_gen, v.g_loss],
            allow_input_downcast=True, mode=mode, givens=v.replace_z)


    def fit(self, X, gen_conditional=None, dis_conditional=None,
            batch_size=128, nb_epoch=100, verbose=0,
            callbacks=None, shuffle=True):
        if callbacks is None:
            callbacks = []
        if gen_conditional is None:
            gen_conditional = []
        if dis_conditional is None:
            dis_conditional = []
        labels = ['d_loss', 'd_real', 'd_gen', 'g_loss']

        def train(model, batch_ids, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}
            ins = [X[batch_ids]]
            for c in gen_conditional + dis_conditional:
                ins.append(c[batch_ids])
            outs = self._train(*ins)
            for key, value in zip(labels, outs):
                batch_logs[key] = value

        assert batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, len(X), batch_size=batch_size, nb_epoch=nb_epoch,
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
                   config['z_shape'],
                   config['num_gen_conditional'],
                   config['num_dis_conditional'],
                   config['num_both_conditional'])

    def get_config(self, verbose=0):
        g_config = self.G.get_config(verbose)
        d_config = self.D.get_config(verbose)
        return {
            'G': g_config,
            'D': d_config,
            'z_shape': self.z_shape,
            'num_gen_conditional': self.num_gen_conditional,
            'num_dis_conditional': self.num_dis_conditional,
            'num_both_conditional': self.num_both_conditional,
        }

    def save(self, directory, overwrite=False):
        os.makedirs(directory, exist_ok=True)
        with open(directory + "/gan.json", "w") as f:
            json.dump(self.get_config(), f)
        self.save_weights(self._weight_fname_tmpl(directory), overwrite)

    def generate(self, *conditional):
        return self._generate(*conditional)[0]

    def debug(self, X, z=None, *conditionals):
        if z is None:
            z = np.random.uniform(-1, 1, self.z_shape)

        labels = ['fake', 'real', 'd_loss', 'd_real', 'd_gen', 'g_loss']
        ins = z + conditionals
        outs = self._generate(ins)
        return DotMap(zip(labels, outs))

    def interpolate(self, x, y):
        z = np.zeros(self.z_shape)
        n = len(z)
        for i in range(n):
            z[i] = x + i/n * (y - x)
        real = np.zeros_like(z)
        outs = self.debug(real, z=z)
        return outs.fake

    def train_batch(self, X, ZD, ZG, k=1):
        for i in range(k):
            self._d_train([X, ZD])
        self._d_train([ZG])
