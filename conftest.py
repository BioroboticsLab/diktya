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

visual_debug = False

import matplotlib   # noqa

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa
import skimage.data   # noqa
import skimage.color  # noqa
import pytest         # noqa
from keras.backend.common import cast_to_floatx   # noqa


TEST_OUTPUT_DIR = "tests_out"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


def plt_save_and_maybe_show(fname):
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, fname))
    if visual_debug:
        plt.show()


@pytest.fixture
def astronaut():
    astronaut = skimage.data.astronaut() / 255.
    astronaut = skimage.color.rgb2gray(astronaut)
    return cast_to_floatx(astronaut[::4, ::4])
