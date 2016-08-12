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


from diktya.plot.latexify import figsize, latexify, savefig_pgf_pdf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import copy


def test_latexify(outdir):
    rc_default = copy.copy(mpl.rcParams)
    assert mpl.rcParams["pgf.texsystem"] != "pdflatex"
    with latexify():
        assert mpl.rcParams["pgf.texsystem"] == "pdflatex"
        assert rc_default != mpl.rcParams
        print(str(outdir))
        fig = plt.figure(figsize=figsize(0.9))
        ax = fig.add_subplot(111)
        x = np.linspace(0, 2*np.pi, 100)
        ax.plot(x, np.sin(x))
        ax.set_xlabel('Frequency')
        ax.set_ylabel('EMA')
        savefig_pgf_pdf(fig, str(outdir.join('latexify')))
        assert outdir.join('latexify.pdf').exists()
        assert outdir.join('latexify.pgf').exists()

    assert rc_default == mpl.rcParams
