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
import random

from colorsys import hsv_to_rgb
import theano
import numpy as np
from diktya.theano.image_transform import upsample, resize_interpolate
import matplotlib.pyplot as plt
from conftest import plt_save_and_maybe_show


def test_upsample(astronaut):
    x = theano.shared(astronaut[np.newaxis, np.newaxis])
    x_up = upsample(resize_interpolate(x, scale=0.5)).eval()
    plt.subplot(121)
    plt.imshow(x.get_value()[0, 0, :])
    plt.subplot(122)
    plt.imshow(x_up[0, 0, :])
    plt_save_and_maybe_show("test_upsample.png")
