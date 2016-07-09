

from pipeline.distributions import Bernoulli, Zeros, Constant, DistributionCollection, \
    Normalization, SubtDivide, UnitIntervalTo, SinCosAngleNorm, Normal, TruncNormal, Uniform, \
    load_from_json, CombineNormalization
import numpy as np


def test_distribution_collection_dtype():
    dists = DistributionCollection({'const': (Constant(5), 2), 'bern': (Bernoulli(), 5)})
    bs = 10
    zeros = np.zeros((bs,), dtype=dists.dtype)
    for name, nb_elems in dists._nb_elems.items():
        assert len(zeros[name][0]) == nb_elems

    zeros = np.zeros((bs,), dtype=dists.norm_dtype)
    for name, nb_elems in dists._nb_elems.items():
        assert len(zeros[name][0]) == nb_elems


def test_distribution_collection_sampling():
    dists = DistributionCollection({'const': (Constant(5), 2), 'bern': (Bernoulli(), 5)})

    bs = 10
    arr = dists.sample(bs)
    assert arr["const"].shape == (bs, 2)
    assert (arr["const"] == 5).all()

    assert arr["bern"].shape == (bs, 5)
    assert (np.logical_or(arr["bern"] == 1, arr["bern"] == 0)).all()


def test_distribution_collection_normalization():
    dists = DistributionCollection({'const': (Constant(5), 2), 'bern': (Bernoulli(), 5)})
    bs = 10
    arr = dists.sample(bs)
    norm_arr = dists.normalize(arr)
    denorm_arr = dists.denormalize(norm_arr)
    assert (arr == denorm_arr).all()


def test_normalizations_are_bijections():
    def is_bijection(norm):
        shape = (100, 20)
        x = np.clip(np.random.normal(0, 1, shape), -np.pi, np.pi)
        x_norm = norm.normalize(x)
        x_denorm = norm.denormalize(x_norm)
        assert x_denorm.shape == shape
        np.testing.assert_allclose(x_denorm, x)

    is_bijection(Normalization())
    is_bijection(SubtDivide(1, 10))
    is_bijection(SubtDivide(100, 0.01))
    is_bijection(SubtDivide(100, 1e5))
    is_bijection(UnitIntervalTo(10, 100))
    is_bijection(UnitIntervalTo(-20, 1))
    is_bijection(SinCosAngleNorm())


def check_dist_default_norm(dist, n=10000):
    arr = dist.sample((n,))
    norm_arr = dist.default_normalization().normalize(arr)
    assert abs(norm_arr.mean()) <= 0.05
    assert norm_arr.std() < 1.5


def test_normal_default_norm():
    check_dist_default_norm(Normal(12, 4))
    check_dist_default_norm(Normal(120, 40))
    check_dist_default_norm(Normal(-12, 0.01))


def test_uniform_default_norm():
    check_dist_default_norm(Uniform(4, 12))
    check_dist_default_norm(Uniform(12, 40))
    check_dist_default_norm(Uniform(-12, 0))
    check_dist_default_norm(Uniform(-1200, -400))


def test_trunc_normal_default_norm():
    # if mean is in the center it could have zero mean normalized
    check_dist_default_norm(TruncNormal(0, 10, 5, 14))

    dist = TruncNormal(-10, 10, 20, 2)
    n = 10000
    arr = dist.sample((n,))
    norm_arr = dist.default_normalization().normalize(arr)
    assert (norm_arr <= 1).all() and (-1 <= norm_arr).all()


def test_bernoulli_default_norm():
    check_dist_default_norm(Bernoulli())


def check_serialization_distribution(dist, n=100000):
    json_str = dist.to_json()
    loaded_dist = load_from_json(json_str)
    assert dist == loaded_dist

    arr = dist.sample((n,))
    arr_loaded = loaded_dist.sample((n,))
    assert abs(arr.mean() - arr_loaded.mean()) <= 0.05
    assert abs(arr.std() - arr_loaded.std()) <= 0.05


def test_constant_serialization():
    check_serialization_distribution(Constant(10), n=10)


def test_zeros_serialization():
    check_serialization_distribution(Zeros(), n=10)


def test_normal_serialization():
    check_serialization_distribution(Normal(0, 0.4))
    check_serialization_distribution(Normal(10, 1))


def test_trunc_normal_serialization():
    check_serialization_distribution(TruncNormal(0, 10, 5, 14), n=100000)
    check_serialization_distribution(TruncNormal(-10, 10, 20, 2), n=100000)


def test_uniform_serialization():
    check_serialization_distribution(Uniform(0, 1))
    check_serialization_distribution(Uniform(-10, 10))


def test_bernoulli_serialization():
    check_serialization_distribution(Bernoulli())


def check_serialization_normalization(norm, n=100):
    json_str = norm.to_json()
    loaded_norm = load_from_json(json_str)
    assert norm == loaded_norm
    arr = np.random.normal(0, 1, (n,))
    np.testing.assert_allclose(norm.normalize(arr), loaded_norm.normalize(arr))


def test_subtdivide_serialization():
    check_serialization_normalization(SubtDivide(10, 50))


def test_unitintervalto_serialization():
    check_serialization_normalization(UnitIntervalTo(0, 1))
    check_serialization_normalization(UnitIntervalTo(-100, 100))


def test_sincos_serialization():
    check_serialization_normalization(SinCosAngleNorm())


def test_combine_normalization_serialization():
    check_serialization_normalization(
        CombineNormalization([SubtDivide(0, 10), UnitIntervalTo(0, 5)]))


def test_distribution_collection_serialization():
    dists = DistributionCollection({'norm': (Normal(5, 2), 2), 'bern': (Bernoulli(), 5)})
    json_str = dists.to_json()
    dists_loaded = load_from_json(json_str)
    assert dists_loaded.dtype == dists.dtype
    assert dists_loaded.norm_dtype == dists.norm_dtype
    n = 50
    arr = dists.sample(n)
    arr_norm = dists.normalize(arr)
    arr_norm_loaded = dists_loaded.normalize(arr)
    for name in dists.names:
        np.testing.assert_allclose(arr_norm[name], arr_norm_loaded[name])

    n = 10000
    arr = dists.sample(n)
    arr_loaded = dists_loaded.sample(n)
    for name in dists.names:
        assert abs(arr[name].mean() - arr_loaded[name].mean()) <= 0.05
        assert abs(arr[name].std() - arr_loaded[name].std()) <= 0.05
