from collections import Iterable
from numbers import Number
from types import FunctionType
import random

import numpy as np
from skimage.transform import warp, resize, AffineTransform


class ImageAugmentation:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, batch):
        xb, yb = batch
        assert (len(xb) == len(yb))
        for idx in range(len(xb)):
            x, y = xb[idx], yb[idx]
            for aug in self.augmentations:
                x, y = aug(x, y)
            xb[idx], yb[idx] = x, y
        return (xb, yb)


class RandomWarpAugmentation:
    """
    Perform random warping transformation on the input data.

    Parameters can be either constant values, a list/tuple containing the lower and upper bounds
    for a uniform distribution of a value generating functions:

    Examples:
        * RandomWarpAugmentation(rotation=0.5 * np.pi)
        * RandomWarpAugmentation(rotation=(-0.25 * np.pi, 0.25 * np.pi))
        * RandomWarpAugmentation(rotation=lambda: np.random.normal(0, np.pi))

    Args:
        fliph_probability: probability of random flips on horizontal (first) axis
        flipv_probability: probability of random flips on vertial (second) axis
        translation: translation of image data among all axis
        rotation: rotation angle in counter-clockwise direction as radians
        scale: scale as proportion of input size
        shear: shear angle in counter-clockwise direction as radians.
        use_diff: whether to use diffeomorphism augmentation (also known as elastic distortion)
        diff_scale: scale parameter of diffeomorphism augmentation
        diff_alpha: alpha parameter of diffeomorphism augmentation
        diff_fix_border: fix_border parameter of diffeomorphism augmentation
        augment_x: whether to augment input data
        augment_y: whether to augment label data
    """
    def __init__(self,
                 fliph_probability=0.5,
                 flipv_probability=0.5,
                 translation=(-5, 5),
                 rotation=(-np.pi / 8, np.pi / 8),
                 scale=(0.9, 1.1),
                 shear=(-0.1 * np.pi, 0.1 * np.pi),
                 use_diff=True,
                 diff_scale=8,
                 diff_alpha=.75,
                 diff_fix_border=False,
                 augment_x=True,
                 augment_y=False):
        self.fliph_probability = self.parse_parameter(fliph_probability)
        self.flipv_probability = self.parse_parameter(flipv_probability)
        self.translation = self.parse_parameter(translation)
        self.rotation = self.parse_parameter(rotation)
        self.scale = self.parse_parameter(scale)
        self.shear = self.parse_parameter(shear)
        self.use_diff = self.parse_parameter(use_diff)
        self.diff_scale = self.parse_parameter(diff_scale)
        self.diff_alpha = self.parse_parameter(diff_alpha)
        self.diff_fix_border = self.parse_parameter(diff_fix_border)
        self.augment_x = augment_x
        self.augment_y = augment_y

    @staticmethod
    def parse_parameter(param):
        if isinstance(param, Iterable):
            if len(param) != 2:
                raise ValueError('lower and upper bound required')
            lower, upper = param
            return lambda: np.random.uniform(lower, upper)
        elif isinstance(param, Number):
            return lambda: param
        elif isinstance(param, FunctionType):
            return param
        else:
            raise TypeError('parameters must either be bounds for a uniform distribution,' +
                            'a single value or a value generating function')

    @staticmethod
    def _center_transform(transform, shape):
        center_transform = AffineTransform(translation=(-shape[0] // 2, -shape[1] // 2))
        uncenter_transform = AffineTransform(translation=(shape[0] // 2, shape[1] // 2))
        return center_transform + transform + uncenter_transform

    @staticmethod
    def _get_frame(shape, line_width, blur=8):
        frame = np.zeros(shape)
        frame[:, :line_width] = 1
        frame[:, -line_width:] = 1
        frame[:line_width, :] = 1
        frame[-line_width:, :] = 1
        return frame

    def _get_random_affine(self, shape):
        t = AffineTransform(scale=(self.scale(), self.scale()),
                            rotation=self.rotation(),
                            shear=self.shear(),
                            translation=self.translation())
        return self._center_transform(t, shape)

    @staticmethod
    def _get_diffeomorphism(shape, scale=30, alpha=1., fix_border=True,
                            random=np.random.uniform):
        """
        Returns a diffeomorphism mapping that can be used wtih ``diff_warp``.

        Args:
            shape (tuple): Shape of the image
            scale: Scale of the diffeomorphism in pixels.
            alpha (float): Intensity of the diffeomorphism. Must be between 0 and 1
            fix_border (boolean): If true the border of the resulting image stay constant.
            random: Function to draw the randomness.  Will be called with
                    ``random(-intensity, intensity, shape)``.
        """
        h, w = shape
        if h == min(h, w):
            dh = int(scale)
            dw = int(scale / h * w)
        else:
            dw = int(scale)
            dh = int(scale / w * h)

        rel_scale = scale / min(h, w)
        intensity = 0.25 * alpha * 1 / rel_scale
        diff_map = np.clip(random(-intensity, intensity, (dh, dw, 2)), -intensity, intensity)
        if fix_border:
            frame = RandomWarpAugmentation._get_frame((dh, dw), 1)
            for i in (0, 1):
                diff_map[:, :, i] = diff_map[:, :, i] * (1 - frame)

        diff_map = resize(diff_map, (h, w, 2), order=3)
        return diff_map

    def get_random_diffeomorphism(self, shape):
        return self._get_diffeomorphism(shape,
                                        scale=self.diff_scale(),
                                        alpha=self.diff_alpha(),
                                        fix_border=self.diff_fix_border())

    def diff_warp(self, diff_map, transform=None, flipv=False, fliph=False):
        def f(xy):
            xi = xy[:, 0].astype(np.int)
            yi = xy[:, 1].astype(np.int)
            off = xy + diff_map[yi, xi]
            if transform is not None:
                off = transform(off)
            if flipv:
                off = off[::-1, :]
            if fliph:
                off = off[:, ::-1]
            return off

        return f

    def __call__(self, x, y):
        if self.augment_x and self.augment_y:
            assert (x.shape == y.shape)

        transform = self._get_random_affine(x.shape)
        diff = self.get_random_diffeomorphism(x.shape)
        warp_f = self.diff_warp(diff, transform,
                                bool(random.random() < self.flipv_probability()),
                                bool(random.random() < self.fliph_probability()))

        if self.augment_x:
            x = warp(x, warp_f, order=3)

        if self.augment_y:
            y = warp(y, warp_f, order=3)

        return x, y


class RandomNoiseAugmentation:
    @staticmethod
    def random_sigma(loc, scale):
        return np.clip(np.random.normal(loc=loc, scale=scale), np.finfo(float).eps, np.inf)

    def apply_noise(self, mat):
        return np.clip(mat + np.random.normal(loc=0., scale=self.sigma(), size=mat.shape), -1, 1)

    def __init__(self,
                 sigma=lambda: RandomNoiseAugmentation.random_sigma(0.03, 0.01),
                 augment_x=True,
                 augment_y=False):
        self.sigma = sigma
        self.augment_x = augment_x
        self.augment_y = augment_y

    def __call__(self, x, y):
        if self.augment_x:
            x = self.apply_noise(x)
        if self.augment_y:
            x = self.apply_noise(x)
        return x, y
