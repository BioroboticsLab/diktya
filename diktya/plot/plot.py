# Copyright 2016 Leon Sixt
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
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rolling_percentile(start_end, values, label=None,
                            batch_window_size=128, percentile=(1, 99),
                            percentile_alpha=0.5,
                            ax=None, color=None):
    start, end = start_end
    if ax is None:
        ax = plt

    rolling_args = {
        'window': batch_window_size,
        'min_periods': 0,
        'center': True
    }
    rolling = pd.Series(values).rolling(**rolling_args)
    means = rolling.mean()
    upper = pd.Series(rolling.apply(np.percentile, args=[percentile[1]]))
    upper = upper.rolling(**rolling_args).mean()
    lower = pd.Series(rolling.apply(np.percentile, args=[percentile[0]]))
    lower = lower.rolling(**rolling_args).mean()
    nb_epochs = end - start
    epoch_labels = np.arange(start, end, nb_epochs / len(means))
    base_line, = ax.plot(epoch_labels, means, label=label, color=color)
    ax.fill_between(epoch_labels, lower, upper,
                    facecolor=base_line.get_color(), alpha=percentile_alpha)
