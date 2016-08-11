import numpy as np
import pytest

from diktya.preprocessing.image import ImageAugmentation, RandomWarpAugmentation, \
    RandomNoiseAugmentation


@pytest.fixture
def batchsize():
    return 16


@pytest.fixture
def shape():
    return (32, 32)


@pytest.fixture
def onedimensional_data_gen(batchsize):
    return lambda: np.random.rand(batchsize, 1)


@pytest.fixture
def twodimensional_data_gen(batchsize, shape):
    return lambda: np.random.rand(batchsize, shape[0], shape[1])


def test_random_warp(onedimensional_data_gen, twodimensional_data_gen):
    X = twodimensional_data_gen()
    y = onedimensional_data_gen()

    aug = ImageAugmentation([RandomWarpAugmentation()])

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)


def test_random_noise(onedimensional_data_gen, twodimensional_data_gen):
    X = twodimensional_data_gen()
    y = onedimensional_data_gen()

    aug = ImageAugmentation([RandomNoiseAugmentation()])

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)


def test_multiple(onedimensional_data_gen, twodimensional_data_gen):
    X = twodimensional_data_gen()
    y = onedimensional_data_gen()

    aug = ImageAugmentation([RandomNoiseAugmentation(),
                             RandomWarpAugmentation()])

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)


def test_label_augmentation(twodimensional_data_gen):
    X = twodimensional_data_gen()
    y = twodimensional_data_gen()

    aug = ImageAugmentation([RandomWarpAugmentation(augment_y=True)])

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)
