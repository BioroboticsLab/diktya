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
from keras.callbacks import Callback
from keras.layers.convolutional import UpSample2D, Convolution2D
import numpy as np
import theano
import theano.tensor as T
from scipy.misc import imsave


class LossPrinter(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        print("#{}-{} ".format(self.epoch, batch), end='')
        for k in self.params['metrics']:
            if k in logs:
                print(" {}: {:.4f}".format(k, float(logs[k])), end='')
        print('')


class Sample(keras.callbacks.Callback):
    def __init__(self, outdir, every_nth_epoch=10):
        super().__init__()
        self.outdir = outdir
        self.every_nth_epoch = every_nth_epoch

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.every_nth_epoch != 0:
            return
        sample = self.model.generate()
        out_dir = os.path.abspath(
            os.path.join(self.outdir, "epoche_{}/".format(epoch)))
        print("Writing {} samples to: {}".format(len(sample), out_dir))
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(sample)):
            outpath = os.path.join(out_dir, str(i) + ".png")

            imsave(outpath,
                   (sample[i]*255).reshape(3, 16, 16).astype(np.uint8))

_upsample_layer = UpSample2D()


def upsample(input):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]
    _upsample_layer.input = input
    return _upsample_layer.get_output(train=False)


_downsample_layer = Convolution2D(1, 1, 2, 2, subsample=(2, 2), border_mode='same')
_downsample_layer.W = theano.shared(np.asarray([[[
    [0.25, 0.25],
    [0.25, 0.25]
]]]))


def downsample(input):
    if type(input) == list:
        assert len(input) == 1
        input = input[0]
    _downsample_layer.input = input
    return _downsample_layer.get_output(train=False)
