
import codecs
import pickle
import os
import tempfile
import warnings
from runipy.notebook_runner import NotebookRunner


def _dump_params(params):
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    pickle.dump(params, tmp_file)
    tmp_file.close()
    return tmp_file.name


def run_notebook(fname, html_report=None, verbose=False, params={}):
    """
    Runs the noteboook with filename ``fname``.

    Args:
        fname (str): Filename of the notebook
        html_report (str): Save an report to this file (as html).
        verbose (bool): Verbose output
        params (dict): Call notebook with this parameters. See ``load_params``.
    """
    with warnings.catch_warnings():
        try:
            from IPython.utils.shimmodule import ShimWarning
            warnings.filterwarnings('error', '', ShimWarning)
        except ImportError:
            class ShimWarning(Warning):
                """Warning issued by iPython 4.x regarding deprecated API."""
                pass
        try:
            # IPython 3
            from IPython.nbconvert.exporters.html import HTMLExporter
            from IPython.nbformat import convert, current_nbformat, reads
        except ShimWarning:
            # IPython 4
            from nbconvert.exporters.html import HTMLExporter
            from nbformat import convert, current_nbformat, reads
        except ImportError:
            # IPython 2
            from IPython.nbconvert.exporters.html import HTMLExporter
            from IPython.nbformat.current import convert, current_nbformat, reads
        finally:
            warnings.resetwarnings()

    if not os.path.exists(fname):
        nbname = os.path.join(os.path.dirname(__file__), 'notebooks', fname + '.ipynb')
        if not os.path.exists(nbname):
            raise Exception("Notebook does not exists: {}."
                            " We also looked for {}".format(fname, nbname))
        fname = nbname
    tmp_fname = _dump_params(params)
    os.environ['NOTEBOOK_PARAMS_FILE'] = tmp_fname

    if verbose:
        print('Running notebook {}'.format(fname))
        if html_report is not None:
            print("Saving report to {}.".format(html_report))
        print('Parameters:')
        for key, value in sorted(params.items()):
            print('{:25}= {}'.format(key, value))

    def print_output(cell, out):
        if out.output_type == 'stream':
            print(out.text, end='')

    notebook = reads(open(fname).read(), 3)
    try:
        nb_runner = NotebookRunner(notebook, output_callback=print_output)
    except TypeError:
        warnings.warn("Normal runipy installation. Cannot display output!"
                      "Use https://github.com/berleon/runipy/tree/callback"
                      " to show outputs of the cells ")
        nb_runner = NotebookRunner(notebook)
    try:
        nb_runner.run_notebook()
    finally:
        if html_report is not None:
            exporter = HTMLExporter()
            output, resources = exporter.from_notebook_node(
                convert(nb_runner.nb, current_nbformat)
            )
            codecs.open(html_report, 'w', encoding='utf-8').write(output)
        os.unlink(tmp_fname)

def load_params(global_dict, default):
    """
    Enables a notebook to receive parameters from ``run_notebook``.
    Injects all parameters into the global scope.

    For example in a notebook ``mynotebook.ipynb``:

    ::
        load_params(globals(), {
            'x': 20
        })
        print(x)
        >> 20

    This can now be called with the run_notebook function:

    ::
        run_notebook("mynotebook.ipynb", params={'x': 42})

    Then ``x`` will have value 42 in this notebook run.

    """
    if 'NOTEBOOK_PARAMS_FILE' in os.environ:
        fname = os.environ['NOTEBOOK_PARAMS_FILE']
        with open(fname, 'rb') as f:
            params = pickle.load(f)
        for key in params.keys():
            if key not in default:
                raise Exception("Got parameter key {} with value {}. "
                                "But not default is given!".format(key, params[key]))
        for key, default_value in default.items():
            if key not in params:
                params[key] = default_value
    else:
        params = default

    for name, value in sorted(params.items()):
        print("{}: {}".format(name, value))
        global_dict[name] = value
