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

from keras.engine.topology import merge
from keras.layers.core import Activation
from keras.utils.layer_utils import layer_from_config

from contextlib import contextmanager
from collections import OrderedDict
import h5py
import json


@contextmanager
def trainable(model, trainable):
    """
    Sets all layers in model to trainable and restores the state afterwards.

    .. warning::

       Be aware, that the keras ``Model.compile`` method is lazy.
       You might want to call ``Model._make_train_function`` to force a compilation.

    Args:
        model: keras model
        trainable (bool): set layer.traiable to this value

    Example:

    .. code:: python

        model = Model(x, y)
        with trainable(model, False):
            # layers of model are now not trainable
            # Do something
            z = model(y)
            [...]

        # now the layers of `model` are trainable again
    """
    trainables = []
    for layer in model.layers:
        trainables.append(layer.trainable)
        layer.trainable = trainable
    yield
    for t, layer in zip(trainables, model.layers):
        layer.trainable = t


def get_layer(keras_tensor):
    """
    Returns the corresponding layer to a keras tensor.
    """
    layer = keras_tensor._keras_history[0]
    return layer


def sequential(layers, ns=None, trainable=True):
    """
    The functional flexible counter part to the keras Sequential model.

    Args:
        layers (list): Can be a arbitrary nested list of layers.
            The layers will be called sequentially. Can contain ``None``'s
        ns (optional str): Namespace prefix of the layers
        trainable (optional bool): set the layer's trainable attribute to this value.

    Returns:
        A function that takes a tensor as input, applies all the layers, and
        returns the output tensor.

    **Simple example:**

    Call a list of layers.

    .. code:: python

        x = Input(shape=(32,))
        y = sequential([
            Dense(10),
            LeakyReLU(0.4),
            Dense(10, activation='sigmoid'),
        ])(x)

        m = Model(x, y)

    **Advanced example:**

    Use a function to construct reoccuring blocks. The ``conv`` functions
    returns a nested list of layers. This allows one to nicely combine and stack
    different building blocks function.

    .. code:: python

        def conv(n, depth=2, f=3, activation='relu'):
            layers = [
                [
                    Convolution2D(n, f, f, border_mode='same'),
                    BatchNormalization(),
                    Activation(activation)
                ]  for _ in range(depth)
            ]
            return layers + [MaxPooling2D()]

        x = Input(shape=(32,))
        y = sequential([
            conv(32),
            conv(64),
            conv(128),
            Flatten(),
            Dense(10, activation='sigmoid'),
        ])(x, ns='classifier')

        m = Model(x, y)

    """
    def flatten(xs):
        for x in xs:
            try:
                for f in flatten(x):
                    if f is not None:
                        yield f
            except TypeError:
                if x is not None:
                    yield x

    for i, l in enumerate(flatten(layers)):
        if ns is not None:
            if '.' not in l.name:
                name = type(l).__name__.lower()
                name = "{:02}_{}".format(i, name)
            l.name = ns + '.' + name
        l.trainable = trainable

    def call(input):
        x = input
        for l in flatten(layers):
            x = l(x)
        return x

    return call


def concat(tensors, axis=1, **kwargs):
    """
    Wrapper around keras merge function.

    Args:
        tensors: list of keras tensors
        axis: concat on  this axis
        kwargs: passed to the merge function

    Returns:
        The concatenated tensor
    """
    if type(tensors) not in (list, tuple):
        return tensors
    elif len(tensors) == 1:
        return tensors[0]

    return merge(tensors, mode='concat', concat_axis=axis,
                 **kwargs)


def rename_layer(keras_tensor, name):
    """
    Renames the layer of the ``keras_tensor``
    """
    layer = get_layer(keras_tensor)
    layer.name = name


def name_tensor(keras_tensor, name):
    """
    Add a layer with this ``name`` that does nothing.

    Usefull to mark a tensor.
    """
    return Activation('linear', name=name)(keras_tensor)


def keras_copy(obj):
    """
    Copies a keras object by using the ``get_config`` method.
    """
    config = obj.get_config()
    del config['name']
    return type(obj)(**config)


def predict_wrapper(func, names):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return OrderedDict(zip(names, out))
    return wrapper


def save_model(model, fname, overwrite=False, attrs={}):
    """
    Saves the weights and the config of ``model`` in the HDF5 file ``fname``.
    The model config is saved as: ``f.attrs["model"] = model.to_json().encode('utf-8')``,
    where ``f`` is the HDF5 file.
    """
    assert 'layer_names' not in attrs
    model.save_weights(fname, overwrite)
    f = h5py.File(fname, 'r+')
    f.attrs['model'] = model.to_json().encode('utf-8')
    for k, v in attrs.items():
        if type(v) == str:
            v = v.encode('utf-8')
        f.attrs[k] = v
    f.close()


def load_model(fname, custom_objects={}):
    """
    Loads the model and weights from ``fname``. Counterpart to :py:func:`save_model`.
    """
    json_config = get_hdf5_attr(fname, 'model').decode('utf-8')
    config = json.loads(json_config)
    model = layer_from_config(config, custom_objects)
    model.load_weights(fname)
    return model

def get_hdf5_attr(fname, attr_name):
    """Returns the toplevel attribute ``attr_name`` of the hdf5 file ``fname``."""
    f = h5py.File(fname, 'r')
    value = f.attrs[attr_name]
    f.close()
    return value
