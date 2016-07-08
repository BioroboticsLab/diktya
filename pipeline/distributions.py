# Copyright 2015 Leon Sixt
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


import numpy as np
import scipy.stats
import json
import copy


def to_radians(x):
    return x / 180. * np.pi


class JsonConvertable:
    def to_config(self):
        return {"name": self.__class__.__name__}

    @classmethod
    def from_config(cls, config):
        args = copy.copy(config)
        del args['name']
        return cls(**args)

    def to_json(self):
        return json.dumps(self.to_config())

    def __eq__(self, other):
        return self.to_config() == other.to_config()

    def __neq__(self, other):
        return not self.__eq__(other)


def get(x, custom_objects={}):
    if type(x) == dict:
        return load_from_config(x, custom_objects)
    else:
        return x


def load_from_config(config, custom_objects={}):
    cls = globals().get(config["name"])
    if cls is None:
        cls = custom_objects.get(config["name"])
    if cls is None:
        raise ValueError("Cannot init class {}. Maybe you have to specify a custom_obejct"
                         .format(config["name"]))
    return cls.from_config(config)


def load_from_json(json_str, custom_objects={}):
    config = json.loads(json_str)
    return load_from_config(config)


class Normalization(JsonConvertable):
    def normalize(self, array):
        return array

    def denormalize(self, array):
        return array


class ConstantNormalization(JsonConvertable):
    def __init__(self, value):
        self.value = value

    def normalize(self, array):
        return np.zeros_like(array)

    def denormalize(self, array):
        return np.ones_like(array)*self.value


class SubtDivide(Normalization):
    def __init__(self, subt, scale):
        assert scale > 0
        self.subt = subt
        self.scale = scale

    def normalize(self, array):
        return (array - self.subt) / self.scale

    def denormalize(self, array):
        return array * self.scale + self.subt

    def to_config(self):
        config = super().to_config()
        config['subt'] = self.subt
        config['scale'] = self.scale
        return config


class UnitIntervalTo(Normalization):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def normalize(self, array):
        return (self.end - self.start) * array + self.start

    def denormalize(self, array):
        return (array - self.start) / (self.end - self.start)

    def to_config(self):
        config = super().to_config()
        config['start'] = self.start
        config['end'] = self.end
        return config


class SinCosAngleNorm(Normalization):
    def normalize(self, array):
        s = np.sin(array)
        c = np.cos(array)
        return np.concatenate([s, c], axis=-1)

    def denormalize(self, array):
        bs, n = array.shape
        assert n % 2 == 0
        s = array[:, :n // 2]
        c = array[:, n // 2:]
        return np.arctan2(s, c)


class CombineNormalization(Normalization):
    def __init__(self, normalizations):
        self.normalizations = [get(n) for n in normalizations]

    def normalize(self, arr):
        x = arr
        for norm in self.normalizations:
            x = norm.normalize(x)
        return x

    def denormalize(self, norm_arr):
        x = norm_arr
        for norm in reversed(self.normalizations):
            x = norm.denormalize(x)
        return x

    def to_config(self):
        config = super().to_config()
        config['normalizations'] = [n.to_config() for n in self.normalizations]
        return config


class Distribution(JsonConvertable):
    def sample(self, shape):
        raise NotImplementedError

    def default_normalization(self):
        return Normalization()


class Zeros(Distribution):
    def sample(self, shape):
        return np.zeros(shape)


class Constant(Distribution):
    def __init__(self, value):
        self.value = value

    def sample(self, shape):
        return self.value*np.ones(shape)

    def default_normalization(self):
        return ConstantNormalization(self.value)

    def to_config(self):
        config = super().to_config()
        config['value'] = self.value
        return config


class Normal(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, shape):
        eps = 1e-7
        return np.random.normal(self.mean, self.std + eps, shape)

    def default_normalization(self):
        return SubtDivide(self.mean, self.std)

    def to_config(self):
        config = super().to_config()
        config['mean'] = self.mean
        config['std'] = self.std
        return config


class TruncNormal(Distribution):
    """
    Normal distribution truncated between [a;b].
    """
    def __init__(self, a, b, mean, std):
        self.a = a
        self.b = b
        self.mean = mean
        self.std = std

    def sample(self, shape):
        def in_distribution_space(x):
            return (x - self.mean) / self.std

        eps = 1e-7
        return scipy.stats.truncnorm.rvs(
            in_distribution_space(self.a),
            in_distribution_space(self.b),
            self.mean, self.std + eps, shape)

    @property
    def length(self):
        return self.b - self.a

    def default_normalization(self):
        return CombineNormalization([SubtDivide(self.a, self.length), UnitIntervalTo(-1, 1)])

    def to_config(self):
        config = super().to_config()
        config['a'] = self.a
        config['b'] = self.b
        config['mean'] = self.mean
        config['std'] = self.std
        return config


class Uniform(Distribution):
    def __init__(self, low, high):
        assert low < high
        self.low = low
        self.high = high

    def sample(self, shape):
        return np.random.uniform(self.low, self.high, shape)

    @property
    def length(self):
        return self.high - self.low

    def default_normalization(self):
        return CombineNormalization([SubtDivide(self.low, self.length), UnitIntervalTo(-1, 1)])

    def to_config(self):
        config = super().to_config()
        config['low'] = self.low
        config['high'] = self.high
        return config


class Bernoulli(Distribution):
    def sample(self, shape):
        return np.random.binomial(1, 0.5, shape)

    def default_normalization(self):
        return UnitIntervalTo(-1, 1)


class DistributionCollection(Distribution, Normalization):
    """
    A collection of multiple distributions:

    Args:
        distributions (dict):  A dictionary where the items have the form:
            * name: distribution
            * name: (distribution, )
            * name: (distribution, nb_elems)
            * name: (distribution, nb_elems, normalization)
            `distribution` must be a subclass of  :class:`.Distribution`.
            The `nb_elems` specify how many elements are drawn from the
            distribution, if omitted it will be set to 1.
            The `normalization` specifies how it is noramlised. It can be
            omitted and will then be set to `dist.default_normalization()`.
    Example:
    ::
        dist = DistributionCollection({
            "x_rotation": Normal(0, 1),
            "y_rotation": (Uniform(-np.pi, np.pi), 1, SinCosAngleNorm()),
            "center": (Normal(0, 2), 2)
        })
        # Sample 10 vectors from the collection
        arr = dist.sample(10)
        # The array is a structured numpy array. The keys are the one from
        # constructure distributions dictionary.
        arr["x_rotation"][0]

        # Normalizes the arr samples, according to the normalisation
        normed = dist.normalize(arr)

        # the normalization/denormalization should be invariant
        assert np.allclose(dist.denormalize(normed), arr)
    """

    def __init__(self, distributions):
        self._dist_nb_elems_norm = {}
        for name, descr in distributions.items():
            if type(descr) not in (list, tuple):
                distribution = get(descr)
                self._dist_nb_elems_norm[name] = (distribution, 1,
                                                  distribution.default_normalization())
            elif len(descr) == 1:
                distribution = get(descr[0])
                self._dist_nb_elems_norm[name] = (distribution, 1,
                                                  distribution.default_normalization())
            elif len(descr) == 2:
                distribution = get(descr[0])
                nb_elems = descr[1]
                self._dist_nb_elems_norm[name] = (distribution, nb_elems,
                                                  distribution.default_normalization())
            elif len(descr) == 3:
                distribution = get(descr[0])
                nb_elems = descr[1]
                norm = get(descr[2])
                self._dist_nb_elems_norm[name] = (distribution, nb_elems, norm)

        self._nb_elems = {n: nb_elems for n, (_, nb_elems, _)
                          in self._dist_nb_elems_norm.items()}
        self._normalizations = {n: norm for n, (_, _, norm)
                                in self._dist_nb_elems_norm.items()}
        self._distributions = {n: dist for n, (dist, _, _)
                               in self._dist_nb_elems_norm.items()}

        self.dtype = [(name, "({},)float32".format(nb_elems))
                      for name, nb_elems in sorted(self._nb_elems.items())]
        self.norm_dtype = [
            (name, "({},)float32".format(nb_elems*len(norm.normalize(np.zeros((1,))))))
            for name, (_, nb_elems, norm) in self._dist_nb_elems_norm.items()]

        self.keys = list(self._nb_elems.keys())

    def sample(self, batch_size):
        arr = np.zeros((batch_size,), dtype=self.dtype)
        for name, dist in self._distributions.items():
            arr[name] = dist.sample(arr[name].shape)
        return arr

    def normalize(self, arr):
        norm_arr = np.zeros((len(arr),), dtype=self.norm_dtype)
        for name, norm in self._normalizations.items():
            norm_arr[name] = norm.normalize(arr[name])
        return norm_arr

    def denormalize(self, norm_arr):
        arr = np.zeros((len(norm_arr),), dtype=self.dtype)
        for name, norm in self._normalizations.items():
            arr[name] = norm.denormalize(norm_arr[name])
        return arr

    def to_config(self):
        config = super().to_config()
        config['distributions'] = {
            name: [dist.to_config(), nb_elems, norm.to_config()]
            for name, (dist, nb_elems, norm) in self._dist_nb_elems_norm.items()
        }
        return config


def examplary_tag_distribution(nb_bits=12):
    return {
        'bits': (Bernoulli(), nb_bits),
        'z_rotation': (Uniform(-np.pi, np.pi), 1, SinCosAngleNorm()),
        'x_rotation': (Zeros(), 1),
        'y_rotation': (Zeros(), 1),
        'center': (Normal(0, 2), 2),
        'radius': (Normal(24, 0.4), 1),
        'inner_ring_radius': (Constant(0.4), 1),
        'middle_ring_radius': (Constant(0.8), 1),
        'outer_ring_radius': (Constant(1), 1),
        'bulge_factor': (Constant(0.4), 1),
        'focal_length': (Constant(2), 1),
    }
