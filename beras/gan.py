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

from dotmap import DotMap
from keras.objectives import mse, binary_crossentropy
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils.theano_utils import ndim_tensor, floatX
import theano
import theano.tensor.shared_randomstreams as T_random
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from keras.models import Sequential, standardize_X, Graph
from keras import optimizers
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
                 z_shapes, num_gen_conditional=0, num_dis_conditional=0,
                 num_both_conditional=0):
        self.num_dis_conditional = num_dis_conditional
        self.G = generator
        self.D = discremenator
        self.num_gen_conditional = num_gen_conditional
        self.num_both_conditional = num_both_conditional
        self.rs = T_random.RandomStreams(1334)

        if type(z_shapes[0]) not in [list, tuple]:
            z_shapes = [z_shapes]
        self.z_shapes = z_shapes

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

    def _get_gen_output(self, gen_conditional, train=True):
        gen_conditional = standardize_X(gen_conditional)
        self.zs = [self.rs.uniform(z, -1, 1) for z in self.z_shapes]
        self.z_labels = ["z_{}".format(i) for i in range(len(self.z_shapes))]
        if len(z_labels) == 1:
            z_labels = ["z"]
        self._set_input(self.G, self.zs + gen_conditional,
                        z_labels + ["cond_{}".format(i)
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

    def compile(self, optimizer_g, optimizer_d, gan_regulizer=None, mode=None, ndim_gen_out=4):
        optimizer_d = optimizers.get(optimizer_d)
        optimizer_g = optimizers.get(optimizer_g)
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

        g_loss, d_loss, reg_updates = gan_regulizer.get_losses(self, g_loss, d_loss)

        g_updates = optimizer_g.get_updates(
            self.G.params, self.G.constraints, g_loss)
        d_updates = optimizer_d.get_updates(
            self.D.params, self.D.constraints, d_loss)
        self._train = theano.function(
            [x_real] + gen_conditional + dis_conditional + both_conditional,
            [d_loss, d_loss_real, d_loss_gen, g_loss],
            updates=d_updates + g_updates + reg_updates,
            allow_input_downcast=True, mode=mode)
        self._generate = theano.function(
            gen_conditional + dis_conditional + both_conditional,
            [self.g_out],
            allow_input_downcast=True,
            mode=mode)
        zs = [ndim_tensor(len(z_shp)) for z_shp in self.z_shapes]
        self._debug = theano.function(
            [x_real] + zs + gen_conditional + dis_conditional + both_conditional,
            [self.g_out, x_real, d_loss, d_loss_real, d_loss_gen, g_loss],
            allow_input_downcast=True, mode=mode, givens=[(self.zs, zs)])

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

    def generate(self, *conditional):
        return self._generate(*conditional)[0]

    def debug(self, X, zs=None, *conditionals):
        if zs is None:
            zs = [np.random.uniform(-1, 1, z_shp) for z_shp in self.z_shapes]

        labels = ['fake', 'real', 'd_loss', 'd_real', 'd_gen', 'g_loss']
        ins = zs + conditionals
        outs = self._generate(ins)
        return DotMap(zip(labels, outs))

    def interpolate(self, x, y):
        assert len(self.z_shapes) == 1
        z = np.zeros(self.z_shapes[0])
        n = len(z)
        for i in range(n):
            z[i] = x + i/n * (y - x)
        real = np.zeros_like(z)
        outs = self.debug(real, zs=[z])
        return outs.fake

    def train_batch(self, X, ZD, ZG, k=1):
        for i in range(k):
            self._d_train([X, ZD])
        self._d_train([ZG])
