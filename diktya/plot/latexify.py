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

"""
Create native looking matplotlib plots
======================================


Adapted from: http://bkanuka.com/articles/native-latex-plots/

"""

import matplotlib as mpl
mpl.use('pgf')   # noqa

from seaborn.rcmod import _RCAesthetics, set_style
import copy
import numpy as np


def figsize(scale=0.5, ratio='golden', textwidth_pt=390.0):
    """
    Returns the figure size as (width, height).

    Args:
        scale (float): Scale of the
        ratio (float, str): Ratio from height to width. Default
            is ``golden`` ratio.
        textwidth_pt (float): Width of the latex page.
            Get this with from LaTeX using ``\the\textwidth``
    """
    inches_per_pt = 1.0/72.27
    if ratio == 'golden':
        ratio = (np.sqrt(5.0)-1.0)/2.0
    elif type(ratio) is str:
        raise Exception("Unknown ratio: {}".format(ratio))

    fig_width = textwidth_pt*inches_per_pt*scale
    fig_height = fig_width*ratio
    return fig_width, fig_height


_default_latexify_style = {
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to
    "font.sans-serif": [],              # inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 9,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
    ]
}


def latexify(rc=None):
    """
    Latexify plots
    """
    style = copy.copy(_default_latexify_style)
    if rc is not None:
        for k, v in rc.items():
            style[k] = v
    return _Latexify(style)


class _Latexify(_RCAesthetics):
    _keys = tuple(_default_latexify_style.keys())
    _set = staticmethod(set_style)


def savefig_pgf_pdf(fig, filename):
    """Saves the figure as ``{filename}.pgf`` and ``{filename}.pgf``."""
    fig.savefig('{}.pgf'.format(filename))
    fig.savefig('{}.pdf'.format(filename))
