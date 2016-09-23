from diktya.random_search import fmin
import numpy as np


def quadratic_function(x):
    return (x - 2) ** 2


def space_function():
    return np.random.uniform(-2, 4)


def test_fmin():
    results = fmin(quadratic_function, space_function, n=50, verbose=1)
    val, space = results[0]
    print(val, space)
    assert val <= 0.1
