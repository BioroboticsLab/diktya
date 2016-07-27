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

from conftest import TEST_OUTPUT_DIR
from diktya.callbacks import SaveModels, LearningRateScheduler, \
    AutomaticLearningRateScheduler, HistoryPerBatch, VisualiseGAN, SampleGAN
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
import keras.backend as K
import os
import filecmp
import pytest
import seaborn  # noqa
from unittest.mock import Mock


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
    real_data = np.random.random((64, 1, 8, 8))
    z = np.random.random((64, 20))
    sampler = SampleGAN(mock.sample, mock.discriminate, z, real_data,
                        should_sample_func=lambda: True)
    logs = {}
    sampler.on_epoch_end(0, logs)
    assert (z == mock.sample.call_args[0][0]).all()
    assert (fake == mock.discriminate.call_args_list[0][0][0]).all()
    assert (real_data == mock.discriminate.call_args_list[1][0][0]).all()
    assert (logs['samples']['fake'] == fake).all()
    assert (logs['samples']['real'] == real_data).all()
    assert (logs['samples']['discriminator_on_fake'] == discriminator_score).all()
    assert (logs['samples']['discriminator_on_real'] == discriminator_score).all()
    assert (logs['samples']['z'] == z).all()


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
    hist.params['metrics'] = ['loss']
    hist.on_epoch_begin(0)
    losses = [[]]
    for i in range(5):
        loss = float(np.random.sample(1))
        hist.on_batch_end(i, logs={'loss': loss})
        losses[-1].append(loss)

    losses.append([])
    hist.on_epoch_begin(1)
    for i in range(5):
        loss = float(np.random.sample(1))
        hist.on_batch_end(i, logs={'loss': loss})
        losses[-1].append(loss)

    losses.append([])
    hist.on_epoch_begin(1)
    for i in range(5):
        loss = float(np.random.sample(1))
        hist.on_batch_end(i, logs={'loss': loss})
        losses[-1].append(loss)

    assert hist.history['loss'] == losses

    hist.on_train_end()
    assert tmpdir.join("history.json").exists()
    assert tmpdir.join("history.png").exists()


def test_history_per_batch_plot():
    hist = HistoryPerBatch()
    hist.params = {}
    hist.params['metrics'] = ['loss']
    hist.on_train_begin(0)
    n = 50
    mean = 1/np.arange(1, n+1)
    std = 1/np.arange(1, n+1)
    for e in range(n):
        hist.on_epoch_begin(e)
        for b in range(100):
            hist.on_batch_begin(b)
            hist.on_batch_end(b, logs={'loss': float(np.random.normal(mean[e], std[e], 1))})
        hist.on_epoch_end(b)

    fig, axes = hist.plot()
    path1 = TEST_OUTPUT_DIR + "/callback_history.png"
    fig.savefig(path1)
    path2 = TEST_OUTPUT_DIR + "/callback_history2.png"
    hist.plot(save_as=path2)
    filecmp.cmp(path1, path2)
