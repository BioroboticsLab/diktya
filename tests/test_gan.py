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
import os
import keras
from keras.callbacks import ModelCheckpoint
import keras.initializations
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, MaxoutDense

from keras.models import Sequential
from keras.optimizers import SGD, Adam
import math
import pytest
from beras.gan import GenerativeAdverserial, LaplacianGenerativeAdverserial
import numpy as np
import matplotlib.pyplot as plt
from beras.util import LossPrinter


def sample_circle(nb_samples):
    center = (0.2, 0.2)
    r = np.random.normal(0.5, 0.1, (nb_samples, ))
    angle = np.random.uniform(0, 2*math.pi, (nb_samples,))
    X = np.zeros((nb_samples, 2))
    X[:, 0] = r*np.cos(angle) + center[0]
    X[:, 1] = r*np.sin(angle) + center[1]
    return X


class Plotter(keras.callbacks.Callback):
    def __init__(self, X, Z, outdir):
        super().__init__()
        self.X = X
        self.Z = Z
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 0:
            self._plot("on_begin_0.png")

    def on_epoch_end(self, epoch, logs={}):
        self._plot("on_end_{}.png".format(epoch))

    def _plot(self, outname):
        Y = self.model.generate(self.Z)
        fig = plt.figure()
        plt.ylim(-1, 1.5)
        plt.xlim(-1, 1.5)
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='.', c='b', alpha=0.2)
        plt.scatter(Y[:, 0], Y[:, 1], marker='.', c='r', alpha=0.2)
        fig.savefig(os.path.join(self.outdir, outname))
        plt.close()

def test_gen_learn_simle_distribution():
    mean = (0.2, 0)
    cov = [[0.5,  0.1],
           [0.2,  0.4]]
    nb_samples = 60000
    # X = np.random.multivariate_normal(mean, cov, (nb_samples,))
    X = sample_circle(nb_samples)

    nb_z = 20
    generator = Sequential()
    generator.add(Dense(nb_z, nb_z, activation='relu', name='d_dense1'))
    generator.add(Dropout(0.5))
    generator.add(Dense(nb_z, nb_z, activation='relu', name='d_dense2'))
    generator.add(Dropout(0.5))
    generator.add(Dense(nb_z, 2, name='d_dense3'))

    discriminator = Sequential()
    discriminator.add(MaxoutDense(2, 20, nb_feature=5))
    discriminator.add(Dropout(0.5))
    discriminator.add(MaxoutDense(20, 20, nb_feature=5))
    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(20, 1, activation='sigmoid', name='d_dense3'))
    gan = GenerativeAdverserial(generator, discriminator)
    gan.compile('adam')
    Z = np.random.uniform(-1, 1, (600, nb_z))
    gan.fit(X, (nb_samples, nb_z), nb_epoch=5, verbose=0,
            callbacks=[LossPrinter(), Plotter(X, Z, "epoches_plot_adam")])
