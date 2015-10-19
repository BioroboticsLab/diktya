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
from keras.models import Sequential, standardize_X
from keras import optimizers
import numpy as np
from beras.models import AbstractModel
from beras.util import upsample

_rs = T_random.RandomStreams(1334)


def get_generator_output(gen, z_shape, train=True, conditional_inputs=[]):
    z = _rs.uniform(z_shape, -1, 1)
    g_input = T.concatenate([z] + conditional_inputs, axis=1)
    gen.layers[0].input = g_input
    return gen.get_output(train)


def stack_laplacian_gans(generators: list, init_z_shape, init_image=None,
                         conditional_inputs=None, train=True):
    """Stacks multiple gan to a laplacian pyramid"""
    scale_shape_up = lambda shp: (shp[0], shp[1], shp[2]*2, shp[3]*2)
    assert len(generators) >= 2

    if conditional_inputs is None:
        conditional_inputs = [[] for i in generators]

    previous_out = []
    if init_image is not None:
        previous_out = [init_image]
    gans_outs = []
    images = []
    z_shape = init_z_shape
    for gen, cond_ins in zip(generators, conditional_inputs):
        cond_ins = standardize_X(cond_ins)
        if previous_out:
            previous_out = upsample(previous_out)
        previous_out = standardize_X(previous_out)
        out = get_generator_output(gen, z_shape, train, previous_out + cond_ins)
        if previous_out:
            image = previous_out[0] + out[0]
            images.append(image)
        gans_outs.append(out)
        previous_out = out
        z_shape = scale_shape_up(z_shape)
    return gans_outs, images, conditional_inputs


class GAN(AbstractModel):
    def __init__(self, generator: Sequential, discremenator: Sequential,
                 z_shape, num_gen_conditional=0, num_dis_conditional=0, num_both_conditional=0):
        self.num_dis_conditional = num_dis_conditional
        self.G = generator
        self.D = discremenator
        self.num_gen_conditional = num_gen_conditional
        self.num_both_conditional = num_both_conditional
        self.rs = T_random.RandomStreams(1334)
        self.z_shape = z_shape

    def _get_gen_output(self, gen_conditional, train=True):
        gen_conditional = standardize_X(gen_conditional)
        z = self.rs.uniform(self.z_shape, -1, 1)
        g_input = T.concatenate([z] + gen_conditional, axis=1)
        self.G.layers[0].input = g_input
        return self.G.get_output(train)

    def _get_dis_output(self, gen_out, x_real, dis_conditional, train=True):
        dis_conditional = standardize_X(dis_conditional)
        d_in = T.concatenate([gen_out, x_real])
        if dis_conditional:
            d_in = T.concatenate([d_in] + dis_conditional, axis=1)
        self.D.layers[0].input = d_in
        return self.D.get_output(train)

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

        x_real = ndim_tensor(len(self.z_shape))
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
