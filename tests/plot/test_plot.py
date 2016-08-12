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

from diktya.plot import plot_rolling_percentile
import numpy as np
import matplotlib.pyplot as plt


def test_plot_rolling_percentile(outdir):
    x = np.linspace(0, 10, 10000)
    y = np.exp(-x) + np.random.normal(0, 1, size=x.shape) * np.exp(-x)
    fig, ax = plt.subplots()
    plot_rolling_percentile((0, 10), y, label='exp + noise', ax=ax)
    fig.savefig(str(outdir.join("plot_rolling_percentile.png")))
