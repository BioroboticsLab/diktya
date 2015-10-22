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
from keras.utils.theano_utils import ndim_tensor

import theano
import theano.tensor.shared_randomstreams as T_random
import theano.tensor as T
from theano.tensor.nnet import binary_crossentropy as bce
from keras.models import Sequential, standardize_X, Graph
from keras import optimizers
import numpy as np
from beras.models import AbstractModel
from beras.util import upsample

_rs = T_random.RandomStreams(1334)


class GAN(AbstractModel):
    ndim = 4

    def __init__(self, generator, discremenator,
                 z_shapes, num_gen_conditional=0, num_dis_conditional=0, num_both_conditional=0):
        self.num_dis_conditional = num_dis_conditional
        self.G = generator
        self.D = discremenator
        self.num_gen_conditional = num_gen_conditional
        self.num_both_conditional = num_both_conditional
        self.rs = T_random.RandomStreams(1334)
        if type(z_shapes) not in [list, tuple]:
            z_shapes = [z_shapes]
        self.z_shapes = z_shapes

    @staticmethod
    def _set_input(model, inputs, labels):
        if type(model) == Sequential:
            input = T.concatenate(inputs, axis=1)
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
        zs = [self.rs.uniform(z, -1, 1) for z in self.z_shapes]
        z_labels = ["z_{}".format(i) for i in range(len(self.z_shapes))]
        if len(z_labels) == 1:
            z_labels = ["z"]
        self._set_input(self.G, zs + gen_conditional,
                        z_labels + ["cond_{}".format(i)
                                    for i in range(len(gen_conditional))])
        return self._get_output(self.G, train)

    def _get_dis_output(self, gen_out, x_real, dis_conditional, train=True):
        dis_conditional = standardize_X(dis_conditional)
        d_in = T.concatenate([gen_out, x_real])
        self._set_input(self.D, [d_in] + dis_conditional,
                        ["input"] +
                        ["cond_{}".format(i)
                         for i in range(len(dis_conditional))])
        return self._get_output(self.D, train)

    def losses(self, x_real, gen_conditional=[], dis_conditional=[],
               both_conditional=[], gen_out=None):
        if gen_out is None:
            gen_out = self._get_gen_output(gen_conditional + both_conditional)
        self.g_out = gen_out
        self.d_out = self._get_dis_output(gen_out, x_real,
                                          dis_conditional + both_conditional)
        batch_size = self.d_out.shape[0]
        d_out_given_g = self.d_out[:batch_size//2]
        d_out_given_x = self.d_out[batch_size//2:]
        clip = lambda x: T.clip(x, 1e-7, 1.0 - 1e-7)
        d_loss_real = bce(clip(d_out_given_x), T.ones_like(d_out_given_x)).mean()
        d_loss_gen = bce(clip(d_out_given_g), T.zeros_like(d_out_given_g)).mean()
        d_loss = d_loss_real + d_loss_gen

        g_loss = bce(clip(d_out_given_g), T.ones_like(d_out_given_g)).mean()
        return g_loss, d_loss, d_loss_real, d_loss_gen

    def compile(self, optimizer_g, optimizer_d, mode=None):
        self.optimizer_d = optimizers.get(optimizer_d)
        self.optimizer_g = optimizers.get(optimizer_g)

        x_real = ndim_tensor(self.ndim)
        gen_conditional = [T.tensor4("gen_conditional_{}".format(i))
                           for i in range(self.num_gen_conditional)]
        both_conditional = [T.tensor4("conditional_{}".format(i))
                            for i in range(self.num_both_conditional)]
        dis_conditional = [T.tensor4("dis_conditional_{}".format(i))
                           for i in range(self.num_dis_conditional)]

        g_loss, d_loss, d_loss_real, d_loss_gen = \
            self.losses(x_real, gen_conditional, dis_conditional, both_conditional)

        g_updates = self.optimizer_g.get_updates(
            self.G.params, self.G.constraints, g_loss)
        d_updates = self.optimizer_d.get_updates(
            self.D.params, self.D.constraints, d_loss)
        self._train = theano.function([x_real] + gen_conditional + dis_conditional + both_conditional,
            [d_loss, d_loss_real, d_loss_gen, g_loss], updates=d_updates + g_updates,
            allow_input_downcast=True, mode=mode)
        self._generate = theano.function(gen_conditional + dis_conditional + both_conditional,
                                         [self.g_out],
                                         allow_input_downcast=True,
                                         mode=mode)

    def fit(self, X, batch_size=128, nb_epoch=100, verbose=0,
            callbacks=None,  shuffle=True):
        if callbacks is None:
            callbacks = []
        labels = ['d_loss', 'd_real', 'd_gen', 'g_loss']

        def train(model, batch_ids, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}
            outs = self._train(X[batch_ids])
            for key, value in zip(labels, outs):
                batch_logs[key] = value

        assert batch_size % 2 == 0, "batch_size must be multiple of two."
        self._fit(train, len(X), batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle, metrics=labels)

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

    def train_batch(self, X, ZD, ZG, k=1):
        for i in range(k):
            self._d_train([X, ZD])
        self._d_train([ZG])
