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


from tests import visual_debug, TEST_OUTPUT_DIR
import os
import keras
import theano
from dotmap import DotMap
import keras.initializations
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten

from keras.models import Sequential, Graph
import math
import pytest

from beras.gan import GAN, sequential_to_gan, gan_binary_crossentropy, \
    add_gan_outputs
import numpy as np
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
    def __init__(self, X, outdir):
        super().__init__()
        self.X = X
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 0:
            self._plot("on_begin_0.png")

    def on_epoch_end(self, epoch, logs={}):
        self._plot("on_end_{}.png".format(epoch))

    def _plot(self, outname):
        import matplotlib.pyplot as plt
        ys = []
        for i in range(32):
            ys.append(self.model.generate())
        Y = np.concatenate(ys)
        fig = plt.figure()
        plt.ylim(-1, 1.5)
        plt.xlim(-1, 1.5)
        plt.scatter(self.X[:, 0], self.X[:, 1], marker='.', c='b', alpha=0.2)
        plt.scatter(Y[:, 0], Y[:, 1], marker='.', c='r', alpha=0.2)
        fig.savefig(os.path.join(self.outdir, outname))
        plt.close()


simple_gan_batch_size = 64
simple_gan_nb_z = 20
simple_gan_nb_out = 2
simple_gan_z_shape = (simple_gan_batch_size, simple_gan_nb_z)


@pytest.fixture()
def simple_gan():
    generator = Sequential()
    generator.add(Dense(simple_gan_nb_z, activation='relu',
                        input_dim=simple_gan_nb_z))
    generator.add(Dense(simple_gan_nb_z, activation='relu'))
    generator.add(Dense(simple_gan_nb_out, activation='sigmoid'))
    discriminator = Sequential()
    discriminator.add(Dense(20, activation='relu', input_dim=2))
    discriminator.add(Dense(1, activation='sigmoid'))
    return sequential_to_gan(generator, discriminator)


def test_gan_learn_simple_distribution():
    def sample_multivariate(nb_samples):
        mean = (0.2, 0)
        cov = [[0.5,  0.1],
               [0.2,  0.4]]
        return np.random.multivariate_normal(mean, cov, (nb_samples,))

    nb_samples = 600
    # X = sample_multivariate(nb_samples)
    X = sample_circle(nb_samples)

    for r in (GAN.Regularizer(), GAN.Regularizer()):
        gan = simple_gan()
        gan.add_gan_regularizer(r)
        gan.compile('adam', 'adam', gan_binary_crossentropy)
        callbacks = [LossPrinter()]
        if visual_debug:
            callbacks.append(Plotter(X, TEST_OUTPUT_DIR + "/epoches_plot"))
        z = np.random.uniform(-1, 1, (len(X), simple_gan_nb_z))
        gan.fit({'real': X, 'z': z}, nb_epoch=1, verbose=0,
                callbacks=callbacks, batch_size={'real': 32, 'z': 96})


@pytest.mark.skipif(reason="No multiple comiples #1")
def test_gan_multiple_compiles(simple_gan):
    simple_gan.compile('adam', 'adam')
    simple_gan.compile('adam', 'adam')


def test_gan_utility_funcs(simple_gan: GAN):
    simple_gan.compile('adam', 'adam', gan_binary_crossentropy)
    xy_shp = simple_gan_z_shape[1:]
    x = np.zeros(xy_shp, dtype=np.float32)
    y = np.zeros(xy_shp, dtype=np.float32)
    simple_gan.interpolate(x, y)

    z_point = simple_gan.random_z_point()
    neighbors = simple_gan.neighborhood(z_point, std=0.05)

    diff = np.stack([neighbors[0]]*len(neighbors)) - neighbors
    assert np.abs(diff).mean() < 0.1


def test_gan_graph():
    g = Graph()
    g.add_input(GAN.z_name, (1, 8, 8))
    g.add_input(GAN.real_name, (1, 8, 8))
    g.add_input("gen_cond", (1, 8, 8))
    g.add_node(Convolution2D(10, 2, 2, activation='relu', border_mode='same'),
               name=GAN.generator_name, inputs=['z', 'gen_cond'], concat_axis=1)
    g.add_node(Convolution2D(5, 2, 2, activation='relu'),
               "dis_conv", inputs=[GAN.generator_name, GAN.real_name],
               concat_axis=1)
    g.add_node(Flatten(), "dis_flatten", input="dis_conv")
    g.add_node(Dense(1, input_dim=20, activation='sigmoid'),
               "dis", input="dis_flatten")
    add_gan_outputs(g, input='dis')
    gan = GAN(g)
    gan.compile('adam', 'adam', gan_binary_crossentropy)
    z_shape = (64, 1, 8, 8)
    gan.generate({'gen_cond': np.zeros(z_shape)}, z_shape)


def test_gan_l2_regularizer():
    reg = GAN.L2Regularizer()
    assert reg.l2_coef.get_value() == 0.
    g_loss = theano.shared(np.cast['float32'](reg.high + 2))
    d_loss = theano.shared(np.cast['float32'](0.))
    gan = DotMap()
    nodes = DotMap()
    nodes.layer.trainable_weights = [theano.shared(np.cast['float32'](1.))]
    gan.get_discriminator_nodes = lambda: nodes
    gan.updates = []
    reg.set_gan(gan)
    reg_g_loss, reg_d_loss = reg(g_loss, d_loss)
    fn = theano.function([], [reg_g_loss, reg_d_loss], updates=gan.updates)
    fn()
    assert reg.l2_coef.get_value() > 0.
    g_loss.set_value(np.cast['float32'](reg.low - 0.1))
    fn()
    fn()
    assert reg.l2_coef.get_value() == 0.
