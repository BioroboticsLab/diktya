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
import json
import copy
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import Callback, CallbackList
import keras.backend as K
import warnings

from diktya.numpy import tile, image_save
from diktya.func_api_helpers import save_model
from diktya.plot.tiles import plt_imshow
from diktya.plot import plot_rolling_percentile


class OnEpochEnd(Callback):
    def __init__(self, func, every_nth_epoch=10):
        self.func = func
        self.every_nth_epoch = every_nth_epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.every_nth_epoch == 0:
            self.func(epoch, logs)


class SampleGAN(Callback):
    """
    Keras callback that provides samples ``on_epoch_end`` to other callbacks.

    Args:
        sample_func: is called with ``z`` and should return fake samples.
        discriminator_func: Should return the discriminator score.
        z: Batch of random vectors
        real_data: Batch of real data
        callbacks: List of callbacks, called with the generated samples.
        should_sample_func (optional): Gets the current epoch and returns
            a bool if we should sample at the given epoch.

    """

    def __init__(self, sample_func, discriminator_func, z, real_data,
                 callbacks,
                 should_sample_func=None):
        self.sample_func = sample_func
        self.discriminator_func = discriminator_func
        self.z = z
        self.real_data = real_data
        self.callback = CallbackList(callbacks)
        if should_sample_func is None:
            def should_sample_func(e):
                return e % 10 == 0 or e <= 15

        self.should_sample_func = should_sample_func

    def sample(self):
        outs = self.sample_func(self.z)
        outs['z'] = self.z
        outs['discriminator_on_fake'] = self.discriminator_func(outs['fake'])
        outs['discriminator_on_real'] = self.discriminator_func(self.real_data)
        outs['real'] = self.real_data
        return outs

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        cbks_logs = copy.copy(logs)
        cbks_logs['samples'] = self.sample()
        self.callback.on_train_begin(cbks_logs)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if self.should_sample_func(epoch):
            cbks_logs = copy.copy(logs)
            cbks_logs['samples'] = self.sample()
            self.callback.on_epoch_end(epoch, cbks_logs)


class VisualiseGAN(Callback):
    """
    Visualise ``nb_samples`` fake images from the generator.

    .. warning::
        Cannot be used as normal keras callback.
        Can only be used as callback for the SampleGAN callback.

    Args:
        nb_samples: number of samples
        output_dir (optional): Save image to this directory. Format is ``{epoch:05d}``.
        show (default: False): Show images as matplotlib plot
        preprocess (optional): Apply this preprocessing function to the generated images.
    """

    def __init__(self, nb_samples, output_dir=None, show=False, preprocess=None):
        self.nb_samples = nb_samples
        self.output_dir = output_dir
        self.show = show
        self.preprocess = preprocess
        if self.preprocess is None:
            self.preprocess = lambda x: x
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_train_begin(self, logs={}):
        self(logs['samples'], os.path.join(self.output_dir, "on_train_begin.png"))

    def call(self, samples):
        return tile(samples['fake'][:self.nb_samples])

    def __call__(self, samples, fname=None):
        selected_samples = {}
        for name, arr in samples.items():
            if len(arr) < self.nb_samples:
                raise Exception("Shoud visualise {} samples. But got only {} for {}".format(
                    self.nb_samples, len(arr), name
                ))
            selected_samples[name] = arr[:self.nb_samples]
        tiles = self.call(selected_samples)
        if self.show:
            plt_imshow(tiles)
            plt.show()
        if fname is not None:
            image_save(fname, tiles)

    def on_epoch_end(self, epoch, logs={}):
        if 'samples' not in logs:
            raise Exception("VisualiseGAN cannot be used as normal keras callback. See docs.")
        if self.output_dir is not None:
            fname = os.path.join(self.output_dir, "{:05d}.png".format(epoch))
        else:
            fname = None
        self(logs['samples'], fname)


class SaveModels(Callback):
    def __init__(self, models, output_dir=None, every_epoch=50,
                 overwrite=True, hdf5_attrs=None):
        """
        Args:
            models: dict with {"name": model}
        """
        self.models = models
        self.every_epoch = every_epoch
        self.overwrite = overwrite
        self.output_dir = output_dir
        if hdf5_attrs is None:
            hdf5_attrs = {}
        self.hdf5_attrs = hdf5_attrs

    def on_epoch_end(self, epoch, log={}):
        epoch = epoch
        if epoch % self.every_epoch == 0 and epoch != 0:
            for name, model in self.models.items():
                fname = name.format(epoch=epoch)
                if self.output_dir is not None:
                    fname = os.path.join(self.output_dir, fname)
                save_model(model, fname, overwrite=self.overwrite, attrs=self.hdf5_attrs)


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
    """
    This callback automatically reduces the learning rate of the `optimizer`.
    If the ``metric`` did not improve by at least the ``min_improvement`` amount in
    the last ``epoch_patience`` epochs, the learning rate of ``optimizer`` will be
    decreased by ``factor``.

    Args:
        optimizer (keras Optimizer): Decrease learning rate of this optimizer
        metric (str): Name of the metric
        min_improvement (float): minimum-improvement
        epoch_patience (int): Number of epochs to wait until the metric decreases
        factor (float): Reduce learning rate by this factor

    """

    def __init__(self, optimizer, metric='loss', min_improvement=0.001,
                 epoch_patience=3, factor=0.25):
        assert hasattr(optimizer, 'lr')
        self.optimizer = optimizer
        self.metric = metric
        self.current_best = np.infty
        self.current_best_epoch = 0
        self.min_improvement = min_improvement
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
        if mean_loss + self.min_improvement <= self.current_best:
            self.current_best = mean_loss
            self.current_best_epoch = epoch

        if epoch - self.current_best_epoch > self.epoch_patience:
            lr = K.get_value(self.optimizer.lr)
            new_lr = lr*self.factor
            self.min_improvement *= self.factor
            K.set_value(self.optimizer.lr, new_lr)
            print()
            print("Reduce learning rate to: {:08f}".format(new_lr))
            self.current_best_epoch = epoch


class HistoryPerBatch(Callback):
    """
    Saves the metrics of every batch.

    Args:
        output_dir (optional str): Save history and plot to this directory.
        extra_metrics (optional list): Also montior this metrics.

    Attributes:
        batch_history: history of every batch.
            Use ``batch_history[metric_name][epoch_idx][batch_idx]`` to index.
        epoch_history: history of every epoch.
            Use ``epoch_history[metric_name][epoch_idx]`` to index.
    """

    def __init__(self, output_dir=None, extra_metrics=None):
        self.batch_history = {}
        self.epoch_history = {}
        self.output_dir = output_dir
        self.extra_metrics = extra_metrics or []

    @staticmethod
    def from_config(batch_history, epoch_history):
        hist = HistoryPerBatch()
        hist.batch_history = batch_history
        hist.epoch_history = epoch_history
        return hist

    @property
    def history(self):
        warnings.warn(
            "history property is deprecated. Use batch_history instead.", DeprecationWarning)
        return self.batch_history

    @property
    def metrics(self):
        """List of metrics to montior."""
        return self.params['metrics'] + self.extra_metrics

    def on_epoch_begin(self, epoch, logs=None):
        for k in self.metrics:
            if k not in self.batch_history:
                self.batch_history[k] = []
            self.batch_history[k].append([])

    def on_batch_end(self, batch, logs={}):
        for k in self.metrics:
            if k not in logs:
                continue
            if k not in self.batch_history:
                self.batch_history[k] = [[]]
            self.batch_history[k][-1].append(float(logs[k]))

    def on_epoch_end(self, epoch, logs={}):
        for metric in self.metrics:
            if metric not in logs:
                continue
            if metric not in self.epoch_history:
                self.epoch_history[metric] = []
            self.epoch_history[metric].append(float(logs[metric]))

    def save(self, fname=None):
        if fname is None and self.output_dir is None:
            raise Exception("fname must be given, if output_dir is not set.")
        if fname is None:
            fname = os.path.join(self.output_dir, "history.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(fname, 'w+') as f:
            json.dump({
                'batch_history': self.batch_history,
                'epoch_history': self.epoch_history,
            }, f)

    def on_train_end(self, logs={}):
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.save()
            fig, _ = self.plot()
            fig.savefig(os.path.join(self.output_dir, "history.png"))
            plt.close(fig)

    def plot(self, metrics=None, fig=None, ax=None, skip_first_epoch=False,
             use_every_nth_batch=1, save_as=None, batch_window_size=128, percentile=(1, 99),
             end=None,
             ):
        """
        Plots the losses and variance for every epoch.

        Args:
            metrics (list): this metric names will be plotted
            skip_first_epoch (bool): skip the first epoch. Use full if the
                first batch has a high loss and brakes the scaling of the loss
                axis.
            fig: matplotlib figure
            ax: matplotlib axes
            save_as (str): Save figure under this path. If ``save_as`` is a
                relative path and ``self.output_dir`` is set, it is appended to
                ``self.output_dir``.
        Returns:
            A tuple of fig, axes
        """
        if fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        if metrics is None:
            metrics = self.metrics
        if end is None:
            end = len(next(iter(self.batch_history.values()))) + 1

        if skip_first_epoch:
            start = 1
        else:
            start = 0

        has_batch_plot = defaultdict(lambda: False)
        for label, epochs in self.batch_history.items():
            if label not in metrics or len(epochs) <= start:
                continue
            values = np.concatenate(epochs[start:end])
            values = values[::use_every_nth_batch]
            if len(values) < 1:
                continue
            plot_rolling_percentile((start, end), values, label=label,
                                    batch_window_size=batch_window_size,
                                    percentile=percentile, ax=ax)
            has_batch_plot[label] = True

        for label, epochs in self.epoch_history.items():
            if label not in metrics or len(epochs) <= start or has_batch_plot[label]:
                continue
            nepochs = len(epochs)
            epoch_labels = np.arange(1, nepochs+1)
            ax.plot(epoch_labels, epochs, label=label)

        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        if save_as:
            if not os.path.isabs(save_as) and self.output_dir:
                path = os.path.join(self.output_dir, save_as)
            else:
                path = save_as
            fig.savefig(path)
        return fig, ax


class SaveModelAndWeightsCheckpoint(Callback):
    '''Similiar to keras ModelCheckpoint, but uses :py:func:`save_model` to save
    the model and weights in one file.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the validation loss will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minization of the monitored. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        hdf5_attrs: Dict of attributes for the hdf5 file.

    '''
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', hdf5_attrs=None):

        super(SaveModelAndWeightsCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.hdf5_attrs = hdf5_attrs or {}

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    save_model(self.model, filepath, overwrite=True, attrs=self.hdf5_attrs)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            save_model(self.model, filepath, overwrite=True, attrs=self.hdf5_attrs)
