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
import shutil
import tempfile

from keras.backend.common import cast_to_floatx
from tests import visual_debug, TEST_OUTPUT_DIR
import os
import keras
import theano
from dotmap import DotMap
from keras.callbacks import ModelCheckpoint
import keras.initializations
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dense, Dropout, MaxoutDense, Flatten

from keras.models import Sequential, Graph
import math
import pytest


from beras.gan import GAN
import numpy as np
from beras.util import LossPrinter
from keras.objectives import mean_squared_error


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
    return GAN(generator, discriminator, simple_gan_z_shape)


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
        gan.compile('adam', 'adam', ndim_gen_out=2, gan_regulizer=r)
        callbacks = [LossPrinter()]
        if visual_debug:
            callbacks.append(Plotter(X, TEST_OUTPUT_DIR + "/epoches_plot"))
        gan.fit(X, nb_epoch=1, verbose=0, callbacks=callbacks)


@pytest.mark.skipif(reason="No multiple comiples #1")
def test_gan_multiple_compiles(simple_gan):
    simple_gan.compile('adam', 'adam', ndim_gen_out=2)
    simple_gan.compile('adam', 'adam', ndim_gen_out=2)


def test_gan_save_load(simple_gan):
    directory = tempfile.mkdtemp()
    simple_gan.save(directory)
    loaded_gan = GAN.load(directory)
    assert tuple(simple_gan.z_shape) == tuple(loaded_gan.z_shape)

    for ps, pl in zip(simple_gan.G.params + simple_gan.D.params,
                      loaded_gan.G.params + loaded_gan.D.params):
        np.testing.assert_allclose(pl.get_value(), ps.get_value())
    shutil.rmtree(directory)


def test_gan_optimize_image(simple_gan):
    loss_fn = lambda t, p: mean_squared_error(t, p).mean()
    simple_gan.compile_optimize_image('adam', loss_fn, ndim_expected=2)
    z = cast_to_floatx(np.random.uniform(-1, 1, simple_gan.z_shape))
    goal = simple_gan.generate()
    nb_iteration = 1000
    optimized_image, optimized_z = simple_gan.optimize_image(
        goal, nb_iteration, z_start=z, verbose=1)
    np.testing.assert_allclose(optimized_image, goal, atol=0.1, rtol=0.1)


def test_gan_utility_funcs(simple_gan: GAN):
    simple_gan.compile('adam', 'adam', ndim_gen_out=2)
    xy_shp = simple_gan.z_shape[1:]
    x = np.zeros(xy_shp, dtype=np.float32)
    y = np.zeros(xy_shp, dtype=np.float32)
    simple_gan.interpolate(x, y)

    z = np.ones(simple_gan.z_shape, dtype=np.float32)
    real = np.zeros(simple_gan.g_output_shape())
    debug_out = simple_gan.debug(real, z)
    for debug_label in ['fake', 'real', 'd_loss', 'd_real', 'd_gen', 'g_loss']:
        assert debug_label in debug_out

    z_point = simple_gan.random_z_point()
    neighbors = simple_gan.neighborhood(z_point, std=0.05)

    diff = np.stack([neighbors[0]]*simple_gan.batch_size) - neighbors
    assert np.abs(diff).mean() < 0.1


def test_gan_graph():
    g1 = Graph()
    g1.add_input("z", (1, 8, 8))
    g1.add_input("cond_0", (1, 8, 8))
    g1.add_node(Convolution2D(10, 2, 2, activation='relu', border_mode='same'),
                name="conv", inputs=['z', 'cond_0'], concat_axis=1)
    g1.add_output("output", input='conv')

    d1 = Sequential()
    d1.add(Convolution2D(5, 2, 2, activation='relu', input_shape=(1, 8, 8)))
    d1.add(Flatten())
    d1.add(Dense(1, input_dim=20, activation='sigmoid'))

    z_shape = (1, 1, 8, 8)
    gan = GAN(g1, d1, z_shape, num_gen_conditional=1)
    gan.compile('adam', 'adam')
    gan.generate(conditionals=[np.zeros(z_shape)])


def test_gan_l2_regularizer():
    reg = GAN.L2Regularizer()
    assert reg.l2_coef.get_value() == 0.
    g_loss = theano.shared(np.cast['float32'](reg.high + 2))
    d_loss = theano.shared(np.cast['float32'](0.))
    gan = DotMap()
    gan.D.params = [theano.shared(np.cast['float32'](1.))]
    reg_g_loss, reg_d_loss, updates = reg.get_losses(gan, g_loss, d_loss)
    fn = theano.function([], [reg_g_loss, reg_d_loss], updates=updates)
    fn()
    assert reg.l2_coef.get_value() > 0.
    g_loss.set_value(np.cast['float32'](reg.low - 0.1))
    fn()
    fn()
    assert reg.l2_coef.get_value() == 0.


def test_conditional_conv_gan():
    g1 = Sequential()
    g1.add(Convolution2D(10, 2, 2, activation='relu', border_mode='same', input_shape=(2, 8, 8)))
    g1.add(UpSampling2D((2, 2)))
    g1.add(Convolution2D(1, 2, 2, activation='sigmoid', border_mode='same'))

    d1 = Sequential()
    d1.add(Convolution2D(5, 2, 2, activation='relu', input_shape=(1, 8, 8)))
    d1.add(MaxPooling2D())
    d1.add(Dropout(0.5))
    d1.add(Flatten())
    d1.add(Dense(10, activation='relu'))
    d1.add(Dropout(0.5))
    d1.add(Dense(1, activation='sigmoid'))
    z_shape = (1, 1, 8, 8)
    gan = GAN(g1, d1, z_shape, num_gen_conditional=1)
    gan.compile('adam', 'adam')
    gan.generate(conditionals=[np.zeros(z_shape)])
