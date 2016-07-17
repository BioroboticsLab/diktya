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
from diktyo.callbacks import SaveModels, LearningRateScheduler, \
    AutomaticLearningRateScheduler, HistoryPerBatch
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
import keras.backend as K
import os
import pytest
import seaborn  # noqa


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


def test_history_per_batch():
    hist = HistoryPerBatch()

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

    fig, axes = hist.plot()
    fig.savefig(TEST_OUTPUT_DIR + "/callback_history.png")
