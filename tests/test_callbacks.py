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

import numpy as np
import os
import h5py
import filecmp
import pytest
import seaborn  # noqa
from unittest.mock import Mock

from diktya.callbacks import SaveModels, LearningRateScheduler, \
    AutomaticLearningRateScheduler, HistoryPerBatch, VisualiseGAN, SampleGAN, \
    SaveModelAndWeightsCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.utils.test_utils import get_test_data
from keras.utils import np_utils
import keras.backend as K


np.random.seed(1337)


def test_SaveModelAndWeightsCheckpoint():
    # adapted from keras tests for ModelCheckpoint

    input_dim = 2
    nb_hidden = 4
    nb_class = 2
    batch_size = 5
    train_samples = 20
    test_samples = 20

    filepath = 'checkpoint.h5'
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    # case 1
    monitor = 'val_loss'
    save_best_only = False
    mode = 'auto'

    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    cbks = [SaveModelAndWeightsCheckpoint(
        filepath, monitor=monitor, save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 2
    mode = 'min'
    cbks = [SaveModelAndWeightsCheckpoint(
        filepath, monitor=monitor, save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 3
    mode = 'max'
    monitor = 'val_acc'
    cbks = [SaveModelAndWeightsCheckpoint(
        filepath, monitor=monitor, save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 4
    save_best_only = True
    cbks = [SaveModelAndWeightsCheckpoint(
        filepath, monitor=monitor, save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 5 - hdf5 attrs
    save_best_only = True
    cbks = [SaveModelAndWeightsCheckpoint(
        filepath, monitor=monitor, save_best_only=save_best_only, mode=mode,
        hdf5_attrs={'test': 'Hello World!'}
    )]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    f = h5py.File(filepath)
    assert f.attrs['test'].decode('utf-8') == 'Hello World!'
    f.close()
    os.remove(filepath)


def test_visualise_gan(tmpdir):
    nb_samples = 64
    vis = VisualiseGAN(nb_samples=nb_samples)

    # can handle to many samples
    samples = {'fake': np.random.random((2*nb_samples, 1, 8, 8))}
    outname = tmpdir.join("vis_gan.png")
    vis(samples, fname=str(outname))
    assert outname.exists()

    # raises if to few samples
    samples = {'fake': np.random.random((nb_samples // 2, 1, 8, 8))}
    outname = tmpdir.join("vis_gan_to_few.png")
    with pytest.raises(Exception):
        vis(samples, fname=str(tmpdir.join("vis_gan_to_few.png")))
    assert not outname.exists()

    samples = {'fake': np.random.random((nb_samples, 1, 8, 8))}
    logs = {'samples': samples}

    vis = VisualiseGAN(nb_samples=nb_samples, output_dir=str(tmpdir))
    vis.on_epoch_end(0, logs)
    assert tmpdir.join("00000.png").exists()


def test_sample_gan():
    mock = Mock()
    fake = np.random.random((64, 1, 8, 8))

    mock.sample.return_value = {'fake': fake}
    discriminator_score = np.random.random((64, 1))
    mock.discriminate.return_value = discriminator_score
    callback = Mock()

    real_data = np.random.random((64, 1, 8, 8))
    z = np.random.random((64, 20))
    sampler = SampleGAN(mock.sample, mock.discriminate, z, real_data,
                        callbacks=[callback],
                        should_sample_func=lambda e: True)
    logs = {}
    sampler.on_epoch_end(0, logs)
    assert callback.on_epoch_end.called
    assert (z == mock.sample.call_args[0][0]).all()
    assert (fake == mock.discriminate.call_args_list[0][0][0]).all()
    assert (real_data == mock.discriminate.call_args_list[1][0][0]).all()
    assert 'samples' not in logs

    cbk_logs = callback.on_epoch_end.call_args[0][1]
    assert (cbk_logs['samples']['fake'] == fake).all()
    assert (cbk_logs['samples']['fake'] == fake).all()
    assert (cbk_logs['samples']['real'] == real_data).all()
    assert (cbk_logs['samples']['discriminator_on_fake'] == discriminator_score).all()
    assert (cbk_logs['samples']['discriminator_on_real'] == discriminator_score).all()
    assert (cbk_logs['samples']['z'] == z).all()


def test_save_models(tmpdir):
    m = Sequential()
    m.add(Dense(3, input_dim=3))
    fname = str(tmpdir) + "/test.hdf5"
    cb = SaveModels({fname: m}, every_epoch=10)
    cb.on_epoch_end(0)
    assert not os.path.exists(fname)
    cb.on_epoch_end(10)
    assert os.path.exists(fname)


def test_save_models_overwrite(tmpdir):
    m = Sequential()
    m.add(Dense(3, input_dim=3))
    fname = str(tmpdir) + "/test.hdf5"
    cb = SaveModels({fname: m}, every_epoch=10, overwrite=False)
    cb.on_epoch_end(10)
    assert os.path.exists(fname)
    with pytest.raises(OSError):
        cb.on_epoch_end(10)


def test_save_models_output_dir(tmpdir):
    m = Sequential()
    m.add(Dense(3, input_dim=3))
    fname = "test.hdf5"
    cb = SaveModels({fname: m}, output_dir=str(tmpdir), every_epoch=10)
    cb.on_epoch_end(10)
    assert os.path.exists(os.path.join(str(tmpdir), fname))


def test_lr_scheduler():
    optimizer = Adam(lr=0.1)
    assert np.allclose(K.get_value(optimizer.lr), 0.1)
    schedule = {20: 0.01, 100: 0.005, 900: 0.0001}
    lr_scheduler = LearningRateScheduler(optimizer, schedule)

    lr_scheduler.on_epoch_end(20)
    assert np.allclose(K.get_value(optimizer.lr), schedule[20])

    lr_scheduler.on_epoch_end(50)
    assert np.allclose(K.get_value(optimizer.lr), schedule[20])

    lr_scheduler.on_epoch_end(100)
    assert np.allclose(K.get_value(optimizer.lr), schedule[100])

    lr_scheduler.on_epoch_end(1000)
    assert np.allclose(K.get_value(optimizer.lr), schedule[100])

    lr_scheduler.on_epoch_end(900)
    assert np.allclose(K.get_value(optimizer.lr), schedule[900])

    lr_scheduler.on_epoch_end(1000)
    assert np.allclose(K.get_value(optimizer.lr), schedule[900])


def test_automatic_lr_scheduler():
    optimizer = Adam(lr=0.1)
    lr_scheduler = AutomaticLearningRateScheduler(optimizer, 'loss')
    lr_scheduler.on_train_begin()
    for i in range(3):
        lr_scheduler.on_epoch_begin(i)
        lr_scheduler.on_batch_end(0, {'loss': 1/(i+1)})
        lr_scheduler.on_epoch_end(i)
    o = 3
    assert np.allclose(K.get_value(optimizer.lr), 0.1)

    for i in range(o, o+5):
        print(i)
        lr_scheduler.on_epoch_begin(i)
        lr_scheduler.on_batch_end(0, {'loss': 1})
        lr_scheduler.on_epoch_end(i)
    print(lr_scheduler.current_best)
    print(lr_scheduler.current_best_epoch)
    assert np.allclose(K.get_value(optimizer.lr), 0.1 * lr_scheduler.factor)


def test_history_per_batch(tmpdir):
    hist = HistoryPerBatch(str(tmpdir))

    hist.params = {}
    hist.params['metrics'] = ['loss', 'val_loss']
    hist.on_epoch_begin(0)
    losses = [[]]
    for i in range(5):
        loss = float(np.random.sample(1))
        hist.on_batch_end(i, logs={'loss': loss})
        losses[-1].append(loss)

    hist.on_epoch_end(0, logs={'loss': 1, 'val_loss': 2})

    losses.append([])
    hist.on_epoch_begin(1)
    for i in range(5):
        loss = float(np.random.sample(1))
        hist.on_batch_end(i, logs={'loss': loss})
        losses[-1].append(loss)
    hist.on_epoch_end(1, logs={'loss': 1, 'val_loss': 2})

    losses.append([])
    hist.on_epoch_begin(2)
    for i in range(5):
        loss = float(np.random.sample(1))
        hist.on_batch_end(i, logs={'loss': loss})
        losses[-1].append(loss)

    hist.on_epoch_end(2, logs={'loss': 1, 'val_loss': 2})

    with pytest.warns(DeprecationWarning):
        assert hist.history['loss'] == losses

    assert hist.epoch_history['loss'] == [1, 1, 1]
    assert hist.epoch_history['val_loss'] == [2, 2, 2]

    hist.on_train_end()
    assert tmpdir.join("history.json").exists()
    assert tmpdir.join("history.png").exists()


def test_history_per_batch_plot(outdir):
    hist = HistoryPerBatch()
    hist.params = {}
    hist.params['metrics'] = ['loss', 'val_loss']
    hist.on_train_begin(0)
    path_cb = str(outdir.join("callback_plot.png"))
    plot_cb = hist.plot_callback(fname=path_cb)
    n = 50
    mean = 1/np.arange(1, n+1)
    std = 1/np.arange(1, n+1)
    for e in range(n):
        hist.on_epoch_begin(e)
        for b in range(100):
            hist.on_batch_begin(b)
            hist.on_batch_end(b, logs={'loss': float(np.random.normal(mean[e], std[e], 1))})
        hist.on_epoch_end(e, logs={'val_loss': float(np.random.normal(mean[e], std[e], 1))})

    plot_cb.on_epoch_end(e)
    fig, axes = hist.plot()
    path1 = str(outdir.join("callback_history.png"))
    fig.savefig(path1)
    path2 = str(outdir.join("callback_history2.png"))
    hist.plot(save_as=path2)
    filecmp.cmp(path1, path2)
    filecmp.cmp(path_cb, path2)
