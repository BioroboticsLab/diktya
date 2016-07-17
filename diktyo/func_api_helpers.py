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

from contextlib import contextmanager
from keras.engine.topology import merge
from keras.layers.core import Activation
import re


@contextmanager
def trainable(model, trainable):
    trainables = []
    for layer in model.layers:
        trainables.append(layer.trainable)
        layer.trainable = trainable
    yield
    for t, layer in zip(trainables, model.layers):
        layer.trainable = t


def keras_copy(obj):
    config = obj.get_config()
    del config['name']
    return type(obj)(**config)


def get_layer(keras_tensor):
    layer = keras_tensor._keras_history[0]
    return layer


def sequential(layers, ns=None, trainable=True):
    def flatten(xs):
        for x in xs:
            try:
                for f in flatten(x):
                    yield f
            except TypeError:
                yield x

    for i, l in enumerate(flatten(layers)):
        l.name.split('_')
        if ns is not None:
            if '.' not in l.name:
                name = re.sub('_\d+$', '', l.name)
                name = "{:02}_{}".format(i, name)
            l.name = ns + '.' + name
        l.trainable = trainable

    def call(input):
        x = input
        for l in flatten(layers):
            x = l(x)
        return x

    return call


def concat(tensors, axis=1, name=None, output_shape=None):
    if type(tensors) not in (list, tuple):
        return tensors
    elif len(tensors) == 1:
        return tensors[0]

    return merge(tensors, mode='concat', concat_axis=axis,
                 name=name, output_shape=output_shape)


def rename_layer(keras_tensor, name):
    layer = keras_tensor._keras_history[0]
    layer.name = name


def name_tensor(keras_tensor, name):
    return Activation('linear', name=name)(keras_tensor)
