
from diktya.cli_ipynb import run_notebook, load_params, _dump_params
import pytest
import os


def get_test_data_fname(name):
    return os.path.join(os.path.dirname(__file__), 'data', name)


@pytest.fixture
def notebook():
    return get_test_data_fname('test.ipynb')


def test_run_notebook(outdir, notebook):
    html = outdir.join('test.html')
    run_notebook(notebook, html_report=str(html), params={'test': 'hello from py.test'})
    assert html.check()
    assert 'hello from py.test' in html.read()


def test_load_params():
    del os.environ['NOTEBOOK_PARAMS_FILE']
    load_params(globals(), {'x': 10})
    assert x == 10  # noqa

    cli_params = {'x': 42}
    os.environ['NOTEBOOK_PARAMS_FILE'] = _dump_params(cli_params)
    load_params(globals(), {'x': 10})
    assert x == 42  # noqa
