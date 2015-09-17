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
import numpy as np
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
    def __init__(self, outdir, z_shape, every_nth_epoch=10):
        super().__init__()
        self.outdir = outdir
        self.z_shape = z_shape
        self.every_nth_epoch = every_nth_epoch

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.every_nth_epoch != 0:
            return
        nb_samples = self.z_shape[0]
        mnist_sample = self.model.generate(
            np.random.uniform(-1, 1, self.z_shape))
        out_dir = os.path.abspath(
            os.path.join(self.outdir, "epoche_{}/".format(epoch)))
        print("Writing {} samples to: {}".format(nb_samples, out_dir))
        os.makedirs(out_dir, exist_ok=True)
        for i in range(nb_samples):
            outpath = os.path.join(out_dir, str(i) + ".png")

            imsave(outpath,
                   (mnist_sample[i]*255).reshape(3, 16, 16).astype(np.uint8))
