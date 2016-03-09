from beras.layers.core import Swap
import numpy as np
import theano


def test_swap():
    layer = Swap(0, 10)
    shape = (4, 32)
    layer.set_input_shape(shape)
    y = layer.get_output(False)
    x = layer.input
    fn = theano.function([x], y)
    arr = np.random.sample(shape).astype(np.float32)
    swaped = arr.copy()
    swaped[:, 0], swaped[:, 10] = swaped[:, 10].copy(), swaped[:, 0].copy()
    assert (fn(arr) == swaped).all()
