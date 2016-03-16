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

import logging
from collections import OrderedDict
from keras.objectives import binary_crossentropy
import keras.backend as K
from beras.layers.core import Split
import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from keras.models import Graph, Sequential
import keras.optimizers
from keras.optimizers import Optimizer
from keras.callbacks import Callback
from beras.models import AbstractModel


def flatten(listOfLists):
    return [el for list in listOfLists for el in list]


def gan_binary_crossentropy(d_out_given_fake_for_gen,
                            d_out_given_fake_for_dis,
                            d_out_given_real):
    d_loss_fake = binary_crossentropy(
        T.zeros_like(d_out_given_fake_for_dis),
        d_out_given_fake_for_dis).mean()
    d_loss_real = binary_crossentropy(
        T.ones_like(d_out_given_real),
        d_out_given_real).mean()
    d_loss = d_loss_real + d_loss_fake
    g_loss = binary_crossentropy(
        T.ones_like(d_out_given_fake_for_gen),
        d_out_given_fake_for_gen).mean()
    return g_loss, d_loss, d_loss_real, d_loss_fake


def gan_linear_losses(d_out_given_fake_for_gen,
                      d_out_given_fake_for_dis, d_out_given_real):
    g_loss = -d_out_given_fake_for_gen.mean()
    d_out_given_fake = d_out_given_fake_for_dis
    d_loss_fake = d_out_given_fake.mean()
    d_loss_fake = d_loss_fake + d_loss_fake**2
    d_loss_real = -d_out_given_real.mean()
    d_loss_real = d_loss_real + d_loss_real**2
    d_loss = d_loss_real + d_loss_fake
    return g_loss, d_loss, d_loss_real, d_loss_fake


def sequential_to_gan(generator: Sequential, discriminator: Sequential,
                      nb_real=32, nb_fake=96,
                      nb_fake_for_gen=64,
                      nb_fake_for_dis=32):
    z_shape = (nb_fake,) + generator.layers[0].input_shape[1:]
    g = Graph()
    g.add_input(GAN.z_name, batch_input_shape=z_shape)
    g.add_node(generator, GAN.generator_name, input=GAN.z_name)

    real_shape = (nb_real,) + g.nodes[GAN.generator_name].output_shape[1:]
    g.add_input(GAN.real_name, batch_input_shape=real_shape)
    g.add_node(discriminator, "discriminator",
               inputs=[GAN.generator_name, "real"], concat_axis=0)
    g.add_node(Split(0, nb_fake_for_gen), GAN.dis_out_given_fake_for_gen,
               input="discriminator", create_output=True)
    g.add_node(Split(nb_fake - nb_fake_for_dis, nb_fake),
               GAN.dis_out_given_fake_for_dis,
               input="discriminator", create_output=True)
    g.add_node(Split(nb_fake, nb_fake+nb_real), GAN.dis_out_given_real,
               input="discriminator", create_output=True)

    gan = GAN(g)
    return gan


def add_gan_outputs(graph: Graph, input="discriminator", fake_for_gen=(0, 64),
                    fake_for_dis=(0, 64),
                    real=(64, 128)):
    graph.add_node(Split(*fake_for_gen),
                   GAN.dis_out_given_fake_for_gen,
                   input=input, create_output=True)
    graph.add_node(Split(*fake_for_dis),
                   GAN.dis_out_given_fake_for_dis,
                   input=input, create_output=True)
    graph.add_node(Split(*real), GAN.dis_out_given_real,
                   input=input, create_output=True)


class GAN(AbstractModel):
    z_name = "z"
    g_out_name = "g_out"
    d_input = 'd_input'
    real_name = "real"
    generator_name = "generator"
    dis_out_given_fake_for_gen = "discriminator_given_fake_for_generator"
    dis_out_given_fake_for_dis = "discriminator_given_fake_for_discriminator"
    dis_out_given_real = "discriminator_given_real"

    class Regularizer(Callback):
        def __call__(self, g_loss, d_loss):
            return g_loss, d_loss

        def set_gan(self, gan):
            self.gan = gan

    class StopRegularizer(Callback):
        def __init__(self, high=1.3):
            self.high = high

        def set_gan(self, gan):
            self.gan = gan

        def __call__(self, g_loss, d_loss):
            d_loss = ifelse(g_loss > self.high,
                            0.*d_loss,
                            d_loss)
            return g_loss, d_loss

    class L2Regularizer(Regularizer):
        def __init__(self, low=0.9, high=1.2, delta_value=5e-5, l2_init=0):
            self.l2_coef = theano.shared(
                np.cast['float32'](l2_init), name="l2_rate")
            self.low = low
            self.high = high
            self.delta = theano.shared(np.cast['float32'](delta_value),
                                       name="l2_rate")

        def _apply_l2_regulizer(self, params, loss):
            l2_loss = T.zeros_like(loss)
            for p in params:
                l2_loss += T.sum(p ** 2) * self.l2_coef
            return loss + l2_loss

        def __call__(self, g_loss, d_loss):
            small_delta = np.cast['float32'](1e-7)
            delta_l2 = ifelse(g_loss > self.high,
                              self.delta,
                              ifelse(g_loss < self.low,
                                     -self.delta,
                                     -small_delta))

            new_l2 = T.maximum(self.l2_coef + delta_l2, 0.)
            updates = [(self.l2_coef, new_l2)]
            self.gan.updates += updates
            params = flatten(
                [n.trainable_weights
                 for n in self.gan.get_discriminator_nodes().values()])
            d_loss = self._apply_l2_regulizer(params, d_loss)
            return g_loss, d_loss

    def __init__(self, graph: Graph):
        self.graph = graph

        assert self.z_name in self.graph.inputs
        assert self.real_name in self.graph.inputs
        assert self.generator_name in self.graph.nodes
        assert self.dis_out_given_fake_for_gen in self.graph.outputs
        assert self.dis_out_given_fake_for_dis in self.graph.outputs
        assert self.dis_out_given_real in self.graph.outputs

        self._gan_regularizers = []
        self.updates = []
        self.metrics = OrderedDict()

        self.logger = logging.getLogger("gan.GAN")
        self.logger.info("Create new GAN")
        self.logger.info("Generator nodes: {}".format(
           ", ".join(iter(self.get_generator_nodes().keys()))))
        self.logger.info("Discriminator nodes: {}".format(
           ", ".join(iter(self.get_discriminator_nodes().keys()))))

    def get_generator_nodes(self):
        return OrderedDict([(name, node)
                            for name, node in self.graph.nodes.items()
                            if name.startswith("gen")])

    def get_discriminator_nodes(self):
        return OrderedDict([(name, node)
                            for name, node in self.graph.nodes.items()
                            if name.startswith("dis")])

    @property
    def z_shape(self):
        return self.graph.inputs[GAN.z_name].input_shape

    def layer_dict(self):
        def process_node(name, node):
            if issubclass(type(node), Graph):
                for n, node in process_graph(node):
                    yield name + "/" + n + "-" + node.name, node
            elif issubclass(type(node), Sequential):
                for n, item in process_list(node):
                    yield name + "#" + n + "-" + item.name,  item
            else:
                yield name, node

        def process_graph(g):
            for name, node in g.nodes.items():
                for n, out in process_node(name, node):
                    yield n, out

        def process_list(l):
            for i, node in enumerate(l.layers):
                for n, out in process_node("{:02d}".format(i), node):
                    yield n, out

        return OrderedDict(sorted(process_graph(self.graph)))

    def debug_dict(self, train=False):
        return OrderedDict([(name, node.get_output(train))
                            for name, node in self.layer_dict().items()])

    def _set_loss_metrics(self, g_loss, d_loss, d_loss_gen, d_loss_real):
        self.metrics['g_loss'] = g_loss
        self.metrics['d_loss'] = d_loss
        self.metrics['d_loss_gen'] = d_loss_gen
        self.metrics['d_loss_real'] = d_loss_real

    def build(self, g_optimizer: Optimizer, d_optimizer: Optimizer,
              loss):
        def get_updates(optimizer, nodes, loss):
            optimizer = keras.optimizers.get(optimizer)

            weights, regularizers, constraints, layer_updates = zip(
                *[n.get_params() for n in nodes])

            for r in regularizers:
                loss = r(loss)

            updates = optimizer.get_updates(
                flatten(weights),
                flatten(constraints), loss)
            return updates + flatten(layer_updates)

        d_out_given_fake_gen = \
            self.graph.outputs[self.dis_out_given_fake_for_gen] \
            .get_output(train=True)
        d_out_given_fake_dis = \
            self.graph.outputs[self.dis_out_given_fake_for_dis] \
            .get_output(train=True)
        d_out_given_real = \
            self.graph.nodes[self.dis_out_given_real].get_output(train=True)

        losses = loss(
            d_out_given_fake_gen, d_out_given_fake_dis, d_out_given_real)
        g_loss, d_loss = losses[:2]
        for r in self._gan_regularizers:
            g_loss, d_loss = r(g_loss, d_loss)

        self._set_loss_metrics(g_loss, d_loss, *losses[2:])

        g_updates = get_updates(
            g_optimizer, self.get_generator_nodes().values(), g_loss)
        d_updates = get_updates(
            d_optimizer, self.get_discriminator_nodes().values(), d_loss)

        ins = [self.graph.inputs[name].input
               for name in self.graph.input_order]
        return {
            'g_updates': g_updates,
            'd_updates': d_updates,
            'ins': ins,
            'g_loss': g_loss,
            'd_loss': d_loss
        }

    def compile(self, g_optimizer: Optimizer, d_optimizer: Optimizer, loss):
        v = self.build(g_optimizer, d_optimizer, loss)
        updates = v['g_updates'] + v['d_updates'] + self.updates
        self._train = K.function(v['ins'],
                                 list(self.metrics.values()),
                                 updates=updates)
        self.compile_generate()

    def compile_generate(self):
        gen_ins = [self.graph.inputs[name].input
                   for name in self.graph.input_order
                   if name.startswith("gen") or name == self.z_name]
        self._generate = K.function(
            gen_ins, self.graph.nodes[self.generator_name].get_output(False))

    def compile_debug(self, train=False):
        ins = [self.graph.inputs[name].input
               for name in self.graph.input_order]
        self._debug = K.function(ins, list(self.debug_dict(train).values()))

    def add_gan_regularizer(self, r):
        r.set_gan(self)
        self._gan_regularizers.append(r)

    def fit(self, data, nb_epoch=100, verbose=0, batch_size=128,
            callbacks=None):
        if type(batch_size) == int:
            batch_size = {name: batch_size for name in data.keys()}
        if callbacks is None:
            callbacks = []

        first_name = next(iter(data.keys()))
        nb_train_sample = len(data[first_name])
        nb_batches = nb_train_sample // batch_size[first_name]

        def train(model, batch_index, batch_logs=None):
            if batch_logs is None:
                batch_logs = {}

            ins = []
            for name in self.graph.input_order:
                b = batch_size[name]
                e = b + batch_size[name]
                ins.append(data[name][b:e])
            outs = self._train(ins)
            for key, value in zip(self.metrics.keys(), outs):
                batch_logs[key] = value

        return self._fit(train, nb_train_sample=nb_train_sample,
                         nb_batches=nb_batches, nb_epoch=nb_epoch,
                         verbose=verbose, callbacks=callbacks, shuffle=False,
                         metrics=list(self.metrics.keys()))

    def fit_generator(self, generator, nb_batches_per_epoch,
                      nb_epoch, batch_size=128, verbose=1, callbacks=[]):
        if callbacks is None:
            callbacks = []

        def train(model, batch_index, batch_logs=None):
            ins = next(generator)
            inputs = [ins[name] for name in self.graph.input_order]
            outs = self._train(inputs)
            for key, value in zip(self.metrics.keys(), outs):
                batch_logs[key] = value

        return self._fit(train,
                         nb_train_sample=batch_size*nb_batches_per_epoch,
                         nb_batches=nb_batches_per_epoch,
                         nb_epoch=nb_epoch, verbose=verbose,
                         callbacks=callbacks, shuffle=False,
                         metrics=list(self.metrics.keys()))

    def generate(self, inputs=None, z_shape=None, nb_samples=None):
        if inputs is None:
            inputs = {}
        if GAN.z_name not in inputs:
            if z_shape is None:
                z_shape = self.z_shape
                if nb_samples:
                    z_shape = (nb_samples, ) + z_shape[1:]
            assert None not in z_shape
            inputs[GAN.z_name] = np.random.uniform(-1, 1, z_shape)
        ins = [inputs[n] for n in self.graph.input_order
               if n in inputs]
        return self._generate(ins)

    def debug(self, inputs={}):
        ins = [inputs[n] for n in self.graph.input_order]
        outs = self._debug(ins)
        return dict(zip(self.layer_dict().keys(), outs))

    def interpolate(self, x, y):
        z = np.zeros(self.z_shape)
        n = len(z)
        for i in range(n):
            z[i] = x + i / n * (y - x)
        return self.generate({GAN.z_name: z})

    def random_z_point(self):
        """returns a random point in the z space"""
        shp = self.z_shape[1:]
        return np.random.uniform(-1, 1, shp)

    def neighborhood(self, z_point=None, std=0.25, n=128):
        """samples the neighborhood of a z_point by adding gaussian noise
         to it. You can control the standard derivation of the noise with std.
          """
        shp = self.z_shape[1:]
        if z_point is None:
            z_point = np.random.uniform(-1, 1, shp)
        z = np.zeros((n,) + shp)
        for i in range(n):
            offset = np.random.normal(0, std, shp)
            z[i] = np.clip(z_point + offset, -1, 1)

        return self.generate({GAN.z_name: z})
