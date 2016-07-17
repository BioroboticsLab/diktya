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

import os

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import Callback
import keras.backend as K

from diktyo.numpy.utils import tile
from scipy.misc import imsave


class VisualiseGAN(Callback):
    def __init__(self, nb_samples, output_dir=None, show=False, preprocess=None):
        self.nb_samples = nb_samples
        self.output_dir = output_dir
        self.show = show
        self.preprocess = preprocess
        if self.preprocess is None:
            self.preprocess = lambda x: x
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def should_visualise(self, i):
        return i % 50 == 0 or \
            (i < 1000 and i % 20 == 0) or \
            (i < 100 and i % 5 == 0) or \
            i < 15

    def on_train_begin(self, logs={}):
        z_shape = self.model.z_shape
        z_shape = (self.nb_samples, ) + z_shape[1:]
        self.z = np.random.uniform(-1, 1, z_shape)

    def __call__(self, fname=None):
        fake = self.model.generate({'z': self.z})
        fake = self.preprocess(fake)
        tiled_fakes = tile(fake)
        if self.show:
            if tiled_fakes.shape[0] == 1:
                plt.imshow(tiled_fakes[0], cmap='gray')
            else:
                plt.imshow(tiled_fakes)
            plt.grid(False)
            plt.axes('off')
            plt.show()
        if fname is not None:
            imsave(fname, np.moveaxis(tiled_fakes, 0, -1))

    def on_epoch_end(self, epoch, logs={}):
        epoch = epoch
        if self.should_visualise(epoch):
            if self.output_dir is not None:
                fname = os.path.join(self.output_dir, "{:05d}.png".format(epoch))
            else:
                fname = None
            self(fname)
            plt.clf()


class SaveModels(Callback):
    def __init__(self, models, output_dir=None, every_epoch=50,
                 overwrite=True):
        """
        models: dict with {"name": model}
        """
        self.models = models
        self.every_epoch = every_epoch
        self.overwrite = overwrite
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, log={}):
        epoch = epoch
        if epoch % self.every_epoch == 0 and epoch != 0:
            for name, model in self.models.items():
                fname = name.format(epoch=epoch)
                if self.output_dir is not None:
                    fname = os.path.join(self.output_dir, fname)
                model.save_weights(fname, self.overwrite)


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler

    Args:
       optimizer (keras Optimizer): schedule the learning rate of this optimizer
       schedule
    """
    def __init__(self, optimizer, schedule):
        assert hasattr(optimizer, 'lr')
        self.optimizer = optimizer
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        epoch = epoch
        if epoch in self.schedule:
            new_value = self.schedule[epoch]
            print()
            print("Setting learning rate to: {}".format(new_value))
            K.set_value(self.optimizer.lr, new_value)


class AutomaticLearningRateScheduler(Callback):
    def __init__(self, optimizer, metric, min_improvment=0.001,
                 epoch_patience=3, factor=0.25):
        assert hasattr(optimizer, 'lr')
        self.optimizer = optimizer
        self.metric = metric
        self.current_best = np.infty
        self.current_best_epoch = 0
        self.min_improvment = min_improvment
        self.epoch_patience = epoch_patience
        self.epoch_log = []
        self.factor = factor

    def on_train_begin(self, logs={}):
        self.current_best = np.infty
        self.current_best_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_log = []

    def on_batch_end(self, batch, logs={}):
        self.epoch_log.append(logs[self.metric])

    def on_epoch_end(self, epoch, logs={}):
        mean_loss = np.array(self.epoch_log).mean()
        if mean_loss + self.min_improvment <= self.current_best:
            self.current_best = mean_loss
            self.current_best_epoch = epoch

        if epoch - self.current_best_epoch > self.epoch_patience:
            lr = K.get_value(self.optimizer.lr)
            new_lr = lr*self.factor
            self.min_improvment *= self.factor
            K.set_value(self.optimizer.lr, new_lr)
            print()
            print("Reduce learning rate to: {:08f}".format(new_lr))
            self.current_best_epoch = epoch


class HistoryPerBatch(Callback):
    """
    Saves the metrics of every batch.

    Attributes:
        history: history of every batch. Use ``history[metric_name][epoch][batch]``
            to index.
    """
    def __init__(self):
        self.history = {}

    def on_epoch_begin(self, epoch, logs=None):
        for k in self.history.keys():
            self.history[k].append([])

    def on_batch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k not in logs:
                continue
            if k not in self.history:
                self.history[k] = [[]]
            self.history[k][-1].append(float(logs[k]))

    def plot(self, fig=None, axes=None):
        """
        Plots the losses and variance for every epoch.

        Args:
            fig: matplotlib figure
            axes: matplotlib axes

        Returns:
            A tuple of fig, axes
        """
        if fig is None and axes is None:
            fig = plt.figure()
        if axes is None:
            axes = fig.add_subplot(111)

        for label, epochs in self.history.items():
            means = []
            for epoch in epochs:
                means.append(np.mean(epoch))
            means = np.array(means)
            axes.plot(np.arange(len(means)), means, label=label)
        axes.set_xlabel('epoch')
        axes.set_ylabel('loss')
        return fig, axes
