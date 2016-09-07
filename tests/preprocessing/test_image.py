import numpy as np
import pytest

from diktya.preprocessing.image import chain_augmentations, RandomWarpAugmentation, \
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
    t = int(3)
    aug = RandomWarpAugmentation(translation=lambda: t)
    Xa = aug(X)
    assert(Xa.shape == X.shape)

    x = X[0]

    id_aug = RandomWarpAugmentation()
    identity = id_aug.get_transformation(x.shape)
    np.testing.assert_allclose(identity(x), x, rtol=1e-5)

    trans = aug.get_transformation(x.shape)
    matrix = trans.affine_transformation.params
    np.testing.assert_allclose(
        matrix,
        [[1, 0, t],
         [0, 1, t],
         [0, 0, 1]], rtol=1e-5)
    x_aug = trans(x)

    # transformations stays constant
    assert (x_aug == trans(x)).all()
    np.testing.assert_allclose(x_aug[:-t, :-t], x[t:, t:], rtol=1e-5)


def test_random_noise(onedimensional_data_gen, twodimensional_data_gen):
    X = twodimensional_data_gen()
    aug = RandomNoiseAugmentation()
    Xa = aug(X)
    assert(Xa.shape == X.shape)


def test_multiple(onedimensional_data_gen, twodimensional_data_gen):
    X = twodimensional_data_gen()
    y = onedimensional_data_gen()

    aug = chain_augmentations(RandomNoiseAugmentation(),
                              RandomWarpAugmentation())

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)


def test_label_augmentation(twodimensional_data_gen):
    X = twodimensional_data_gen()
    y = X.copy()

    aug = chain_augmentations(RandomWarpAugmentation(), augment_y=True)

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)
    assert (Xa == ya).all()
