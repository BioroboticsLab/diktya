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


from beras.callbacks import SaveModels
from keras.models import Sequential
from keras.layers.core import Dense
import os
import pytest


def test_save_models(tmpdir):
    m = Sequential()
    m.add(Dense(3))
    fname = str(tmpdir) + "/test.hdf5"
    cb = SaveModels({fname: m}, every_epoch=10)
    cb.on_epoch_end(0)
    assert not os.path.exists(fname)
    cb.on_epoch_end(9)
    assert os.path.exists(fname)


def test_save_models_overwrite(tmpdir):
    m = Sequential()
    m.add(Dense(3))
    fname = str(tmpdir) + "/test.hdf5"
    cb = SaveModels({fname: m}, every_epoch=10, overwrite=False)
    cb.on_epoch_end(9)
    assert os.path.exists(fname)
    with pytest.raises(OSError):
        cb.on_epoch_end(9)


def test_save_models_output_dir(tmpdir):
    m = Sequential()
    m.add(Dense(3))
    fname = "test.hdf5"
    cb = SaveModels({fname: m}, output_dir=str(tmpdir), every_epoch=10)
    cb.on_epoch_end(9)
    assert os.path.exists(os.path.join(str(tmpdir), fname))
