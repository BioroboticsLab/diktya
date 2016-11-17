# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
.. note::

    The functions in this module are espacially usefull together
    with the :py:func:`sequential
    <diktya.func_api_helpers.sequential>` function.
"""


from keras.layers import Layer, merge
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from diktya.func_api_helpers import sequential


def get_activation(activation):
    if type(activation) is str:
        return Activation(activation)
    elif issubclass(type(activation), Layer):
        return activation
    else:
        raise Exception("Did not understand activation: {}".format(activation))


def conv2d_block(n, filters=3, depth=1, border='same', activation='relu',
                 batchnorm=True, pooling=None, up=False, subsample=1):
    """
    2D-Convolutional block consisting of possible muliple repetitions of ``Convolution2D``,
    ``BatchNormalization``, and ``Activation`` layers and can be finished by
    either a ``MaxPooling2D``, a ``AveragePooling2D`` or a ``UpSampling2D`` layer.



    Args:
        n: number of filters of the convolution layer
        filters: shape of the filters are ``(filters, filters)``
        depth: repeat the convolutional, batchnormalization, activation blocks
            this many times
        border: border_mode of the Convolution2D layer
        activation: name or activation or a advanced Activation layer.
        batchnorm: use batchnorm layer if true. If it is an integer it indicates
            the batchnorm mode.
        pooling: if given, either ``max`` or `avg` for MaxPooling2D or
            AveragePooling2D
        up: if true, use a UpSampling2D as last layer. Cannot be true if also
            pooling is given.

    Returns:
        A nested list containing the layers.
    """
    def get_pooling():
        if pooling == 'max':
            return MaxPooling2D()
        elif pooling == 'avg':
            return AveragePooling2D()
        elif pooling is None:
            return
        else:
            raise Exception("Expected `max`,`avg` or `None` for pooling argument."
                            " Got: {}.".format(pooling))

    def get_batchnorm():
        if type(batchnorm) is int:
            return BatchNormalization(mode=batchnorm, axis=1)
        elif type(batchnorm) is bool:
            if batchnorm:
                return BatchNormalization(mode=2, axis=1)
            else:
                return
        else:
            raise Exception("Did not understand batchnorm argument: {}"
                            .format(batchnorm))

    def get_upsampling():
        if up:
            return UpSampling2D()

    def get_conv2d(d):
        if d == 0:
            return Convolution2D(n, filters, filters, border_mode=border,
                                 subsample=(subsample, subsample))
        else:
            return Convolution2D(n, filters, filters, border_mode=border)

    if sum([pooling is not None, up, subsample != 1]) > 1:
        raise Exception("You want to use pooling: {}, up: {}, and {}."
                        "But you can only use one at a time.".format(pooling, up, subsample != 1))

    return [
        [
            get_conv2d(d),
            get_batchnorm(),
            get_activation(activation),
        ]
        for d in range(depth)
    ] + [get_pooling(), get_upsampling()]


def resnet(n, filters=3, activation='relu'):
    """
    A ResNet block. If the number of filter maps is not equal to ``n``,
    a :py:func:`conv2d_block` with ``n`` filter maps is added.

    Args:
        n: number of filters
        filters: size of the conv filters

    Returns:
        A function that takes a keras tensor as input and runs the resnet block
    """
    def wrapper(x):
        shape = x._keras_shape
        if shape[1] != n:
            x = sequential(conv2d_block(n, filters, depth=1, activation=activation))(x)

        f = sequential(conv2d_block(n, filters, depth=2, activation=activation))
        return merge([x, f(x)], mode='sum')
    return wrapper
