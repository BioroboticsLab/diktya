import numpy as np
import pytest

from diktya.preprocessing.image import chain_augmentations, WarpAugmentation, \
    NoiseAugmentation, random_std, ChannelScaleShiftAugmentation


@pytest.fixture
def batchsize():
    return 16


@pytest.fixture
def shape2d():
    return (32, 32)


@pytest.fixture
def shape3d():
    return (3, 32, 32)


@pytest.fixture
def data_gen1d(batchsize):
    return lambda: np.random.rand(batchsize, 1)


@pytest.fixture
def data_gen2d(batchsize, shape2d):
    return lambda: np.random.rand(batchsize, shape2d[0], shape2d[1])


@pytest.fixture
def data_gen3d(batchsize, shape3d):
    return lambda: np.random.rand(batchsize, *shape3d)


def test_channel_scale_shift(data_gen3d):
    x = data_gen3d()
    aug = ChannelScaleShiftAugmentation((0.8, 1.2), (-0.3, 0.3))
    x_aug = aug(x)
    assert x_aug.max() <= aug.max
    assert aug.min <= x_aug.min()

    trans = aug.get_transformation(x.shape[1:])
    assert len(trans.scale) == x.shape[1]
    assert len(trans.shift) == x.shape[1]

    with pytest.raises(Exception):
        aug.get_transformation((1, 1, 1, 1))
    with pytest.raises(Exception):
        aug.get_transformation((1, 1))


def test_warp(data_gen1d, data_gen2d):
    X = data_gen2d()
    t = int(3)
    aug = WarpAugmentation(translation=lambda: t)
    Xa = aug(X)
    assert(Xa.shape == X.shape)

    x = X[0]

    id_aug = WarpAugmentation()
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


def test_noise_augmentation(data_gen1d, data_gen2d):
    X = data_gen2d()
    mean, std = 0.03, 0.01
    aug = NoiseAugmentation(random_std(mean, std))
    Xa = aug(X)

    stds = [aug.std() for _ in range(1000)]
    assert abs(np.std(stds) - std) <= 0.001
    assert abs(np.mean(stds) - mean) <= 0.001

    trans = aug.get_transformation((10000,))
    assert abs(trans.noise.std() - trans.std) <= 0.001
    assert abs(trans.noise.mean()) <= 0.001
    assert(Xa.shape == X.shape)


def test_multiple(data_gen1d, data_gen2d):
    X = data_gen2d()
    y = data_gen1d()

    aug = chain_augmentations(NoiseAugmentation(),
                              WarpAugmentation())

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)


def test_label_augmentation(data_gen2d):
    X = data_gen2d()
    y = X.copy()

    aug = chain_augmentations(WarpAugmentation(), augment_y=True)

    Xa, ya = aug((X, y))
    assert(Xa.shape == X.shape)
    assert(ya.shape == y.shape)
    assert (Xa == ya).all()
