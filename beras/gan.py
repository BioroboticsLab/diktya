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


class GenerativeAdverserial(Model):
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
        updates_D = self.optimizer.get_updates(
            self.D.params, self.D.constraints, loss_train_D)
        self._train_D = theano.function(
            [x_train, z_train], [loss_train_D, d_loss_real, d_loss_gen], updates=updates_D,
            allow_input_downcast=True, mode=mode)
        self._evaluate_D = theano.function(
            [x_train, z_train], [loss_train_D, d_loss_real, d_loss_gen],
            allow_input_downcast=True, mode=mode)
        loss_train_G = bce(d_out_given_g, T.ones_like(d_out_given_g)).mean()
        updates_G = self.optimizer.get_updates(
            self.G.params, self.G.constraints, loss_train_G)
        self._evaluate_G = theano.function([z_train], [loss_train_G],
                                           allow_input_downcast=True, mode=mode)
        self._train_G = theano.function([z_train], [loss_train_G], updates=updates_G,
                                        allow_input_downcast=True, mode=mode)
        self._generate = theano.function(
            [self.G.get_input(train=True)], [self.G.get_output(train=True)],
            allow_input_downcast=True, mode=mode)

    def print_svg(self):
        theano.printing.pydotprint(self._train_G, outfile="train_g.png")
        theano.printing.pydotprint(self._train_D, outfile="train_d.png")
        theano.printing.pydotprint(self._evaluate_D, outfile="evaluate_d.png")
        theano.printing.pydotprint(self._evaluate_G, outfile="evaluate_g.png")

    def fit(self, X, z_shape=None, batch_size=128, nb_epoch=100,
            verbose=0, callbacks=None,  shuffle=True, show_generator_accuracy=False):
        if callbacks is None:
            callbacks = []

        X = standardize_X(X)
        ZD = standardize_X(np.random.sample(z_shape))
        ZG = standardize_X(np.random.sample(z_shape))
        evaluate_labels = ['err_D_on_x', 'err_D_on_gen', 'err_G']
        D_out_labels = ['D_loss', 'D_real', 'D_gen']
        G_out_labels = ['G_loss']
        ins_D = X + ZD
        ins_G = ZG
        train_D = True
        train_G = True
        loss_margin = 0.2
        nb_train_sample = X[0].shape[0]
        index_array = np.arange(nb_train_sample)
        metrics = D_out_labels + G_out_labels + evaluate_labels
        history = cbks.History()
        if verbose:
            callbacks = [history, cbks.BaseLogger()] + callbacks
        else:
            callbacks = [history] + callbacks
        callbacks = cbks.CallbackList(callbacks)

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': False,
            'metrics': metrics,
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            ZD[0] = np.random.normal(0, 0.8, z_shape)
            ZG[0] = np.random.normal(0, 0.8, z_shape)
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            next_evaluate_index = 0
            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    ins_D_batch = slice_X(ins_D, batch_ids)
                    ins_G_batch = slice_X(ins_G, batch_ids)
                except TypeError as err:
                    print('TypeError while preparing batch. \
                        If using HDF5 input data, pass shuffle="batch".\n')
                    raise
                skip_evaluation = 0
                batch_logs = {}
                if batch_index == next_evaluate_index:
                    err_D = self._evaluate_D(*ins_D_batch)
                    err_D_on_gen = err_D[2]
                    batch_logs['err_D_on_x'] = err_D[1]
                    batch_logs['err_D_on_gen'] = err_D_on_gen
                    err_G = self._evaluate_G(*ins_G_batch)[0]
                    batch_logs['err_G'] = err_G
                    if err_G > err_D_on_gen + loss_margin:
                        train_G = True
                        train_D = False
                        skip_evaluation = 4
                    elif err_D_on_gen + loss_margin > err_G > err_D_on_gen:
                        train_G = True
                        train_D = True
                    elif err_D_on_gen > err_G:
                        train_G = True
                        train_D = True
                    if err_D_on_gen > 1.2:
                        train_D = True
                        skip_evaluation = 0
                    next_evaluate_index = batch_index + skip_evaluation + 1

                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                if train_D:
                    outs_D = self._train_D(*ins_D_batch)
                    for l, o in zip(D_out_labels, outs_D):
                        batch_logs[l] = o
                if train_G:
                    outs_G = self._train_G(*ins_G_batch)
                    for l, o in zip(G_out_labels, outs_G):
                        batch_logs[l] = o
                callbacks.on_batch_end(batch_index, batch_logs)


            callbacks.on_epoch_end(epoch)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return history

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
            self._train_D([X, ZD])
        self._train_D([ZG])


