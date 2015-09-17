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
import theano.tensor as T
from theano.tensor.nnet import binary_crossentropy as bce
from keras.models import Sequential, Model, standardize_X, standardize_weights, \
    batch_shuffle, make_batches, slice_X, weighted_objective
import keras.callbacks as cbks
from keras import optimizers
import numpy as np
from beras.models import AbstractModel


class GANTrainer(cbks.Callback):
    def __init__(self, X, z_shape, d_train, g_train, d_evaluate, g_evaluate):
        super().__init__()
        self.next_evalutation = 0
        self.d_train = d_train
        self.g_train = g_train
        self.g_evaluate = g_evaluate
        self.d_evaluate = d_evaluate
        self.X = X
        self.z_shape = z_shape
        self.evaluate_labels = ['eval_d_real', 'eval_d_gen', 'eval_g_loss']
        self.D_out_labels = ['d_loss', 'd_real', 'd_gen']
        self.G_out_labels = ['g_loss']
        self.g_optimize = True
        self.d_optimize = True

    def metrics(self):
        return self.evaluate_labels + self.D_out_labels + self.G_out_labels

    def on_epoch_begin(self, epoch, logs={}):
        self.ZD = standardize_X(np.random.sample(self.z_shape))
        self.ZG = standardize_X(np.random.sample(self.z_shape))
        self.g_ins = self.ZG
        self.d_ins = standardize_X(self.X) + self.ZD

    def _evaluate(self, d_ins, g_ins, batch_logs):
        loss_margin = 0.3
        d_losses = self.d_evaluate(*d_ins)
        d_loss_real = d_losses[0]
        d_loss_gen = d_losses[2]
        batch_logs['eval_d_real'] = d_losses[1]
        batch_logs['eval_d_gen'] = d_loss_gen
        g_loss = self.g_evaluate(*g_ins)[0]
        batch_logs['eval_g_loss'] = g_loss
        if d_loss_gen > d_loss_real + loss_margin:
            self.g_optimize = False
            self.d_optimize = True
        elif d_loss_real + loss_margin >= d_loss_gen > d_loss_real:
            self.g_optimize = True
            self.d_optimize = False
        elif d_loss_gen <= d_loss_real:
            self.g_optimize = True
            self.d_optimize = True
        if d_loss_real > 0.7:
            self.d_optimize = True

    def train(self, model, batch_ids, batch_index, batch_logs=None):
        if batch_logs is None:
            batch_logs = {}

        d_ins_batch = slice_X(self.d_ins, batch_ids)
        g_ins_batch = slice_X(self.g_ins, batch_ids)
        self._evaluate(d_ins_batch, g_ins_batch, batch_logs)
        if self.d_optimize:
            outs_D = self.d_train(*d_ins_batch)
            for l, o in zip(self.D_out_labels, outs_D):
                batch_logs[l] = o
        if self.g_optimize:
            outs_G = self.g_train(*g_ins_batch)
            for l, o in zip(self.G_out_labels, outs_G):
                batch_logs[l] = o


class GenerativeAdverserial(AbstractModel):
    def __init__(self, generator: Sequential, detector: Sequential):
        super().__init__()
        self.G = generator
        self.D = detector

    def compile(self, optimizer, mode=None):
        self.optimizer = optimizers.get(optimizer)

        # reset D's input to X
        ndim = self.D.layers[0].input.ndim
        self.D.layers[0].input = ndim_tensor(ndim)
        x_train = self.D.get_input(train=True)
        d_out_given_x = self.D.get_output(train=True)

        # set D's input to G's output
        self.D.layers[0].input = self.G.get_output(train=True)
        d_out_given_g = self.D.get_output(train=True)
        z_train = self.G.get_input(train=False)

        d_loss_real = bce(d_out_given_x, T.ones_like(d_out_given_x)).mean()
        d_loss_gen = bce(d_out_given_g, T.zeros_like(d_out_given_g)).mean()

        loss_train_D = d_loss_real + d_loss_gen
        d_updates = self.optimizer.get_updates(
            self.D.params, self.D.constraints, loss_train_D)
        self._d_train = theano.function(
            [x_train, z_train], [loss_train_D, d_loss_real, d_loss_gen], updates=d_updates,
            allow_input_downcast=True, mode=mode)
        self._d_evaluate = theano.function(
            [x_train, z_train], [loss_train_D, d_loss_real, d_loss_gen],
            allow_input_downcast=True, mode=mode)
        g_loss = bce(d_out_given_g, T.ones_like(d_out_given_g)).mean()
        g_updates = self.optimizer.get_updates(
            self.G.params, self.G.constraints, g_loss)
        self._g_evaluate = theano.function([z_train], [g_loss],
                                           allow_input_downcast=True, mode=mode)
        self._g_train = theano.function([z_train], [g_loss], updates=g_updates,
                                        allow_input_downcast=True, mode=mode)
        self._generate = theano.function(
            [self.G.get_input(train=True)], [self.G.get_output(train=True)],
            allow_input_downcast=True, mode=mode)

    def print_svg(self):
        theano.printing.pydotprint(self._g_train, outfile="train_g.png")
        theano.printing.pydotprint(self._d_train, outfile="train_d.png")
        theano.printing.pydotprint(self._d_evaluate, outfile="evaluate_d.png")
        theano.printing.pydotprint(self._g_evaluate, outfile="evaluate_g.png")

    def fit(self, X, z_shape, batch_size=128, nb_epoch=100, verbose=0,
            callbacks=None,  shuffle=True):
        trainer = GANTrainer(X, z_shape,
                             d_train=self._d_train, d_evaluate=self._d_evaluate,
                             g_train=self._g_train, g_evaluate=self._g_evaluate)
        if callbacks is None:
            callbacks = []
        callbacks.append(trainer)
        self._fit(trainer.train, len(X), batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, callbacks=callbacks, shuffle=shuffle, metrics=trainer.metrics())

    def load_weights(self, fname):
        self.G.load_weights(fname.format("generator"))
        self.D.load_weights(fname.format("detector"))

    def save_weights(self, fname, overwrite=False):
        self.G.save_weights(fname.format("generator"), overwrite)
        self.D.save_weights(fname.format("detector"), overwrite)

    def generate(self, Z):
        return self._generate(Z)[0]

    def train_batch(self, X, ZD, ZG, k=1):
        for i in range(k):
            self._d_train([X, ZD])
        self._d_train([ZG])


