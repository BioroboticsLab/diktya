from collections import Iterable
from numbers import Number
from types import FunctionType
import random

import numpy as np
from skimage.transform import warp, resize, AffineTransform


def chain_augmentations(*augmentations, augment_x=True, augment_y=False):
    """
    Chain multiple augmentations.

    Example:

    .. code-block:: python

        aug = chain_augmentations(RandomNoiseAugmentation(),
                                  RandomWarpAugmentation())

        Xa, ya = aug((X, y))

    Args:
        augmentations (Augmentation): the leftest augmentations is applied first
        augment_x (bool): should augment data X
        augment_y (bool): should augment label y

    Returns:
        A function that takes a minibatch as input and applies the augmentations.
        The minibatch can either be a numpy array or tuple of (X, y)
        where X are the data and y the labels.

    """
    def wrapper(batch):
        if type(batch) == tuple:
            X, Y = batch
            if len(X) != len(Y):
                raise Exception("Got tuple but arrays have different size. "
                                "Got X= {} and y={}".format(X.shape, Y.shape))
        elif type(batch) == np.ndarray:
            X = batch
            Y = None
        X_aug = []
        Y_aug = []
        for i in range(len(X)):
            x = X[i]
            if Y is not None:
                y = Y[i]
            else:
                y = None
            for aug in augmentations:
                transformation = aug.get_transformation(X[i].shape)
                if augment_x:
                    x = transformation(x)
                if augment_y:
                    y = transformation(y)

            X_aug.append(x)
            if y is not None:
                Y_aug.append(y)
        X_aug = np.stack(X_aug)
        if Y_aug is not None:
            return X_aug, np.stack(Y_aug)
        elif Y is not None:
            return X_aug, Y
        else:
            return X_aug

    return wrapper


class Augmentation:
    """
    Augmentation super class. Subclasses must implement the ``get_transformation``
    method.

    """
    def get_transformation(self, shape):
        """
        Returns a transformation. A transformation can be a function
        or other callable object (``__call__``).  It must map image
        to the augmented image. See :py:meth:`.RandomWarpAugmentation.transformation`
        for an example.
        """
        raise NotImplementedError()

    def __call__(self, batch):
        """
        Applies random augmentations to each sample in the batch.
        The first axis of the batch must be the samples.

        Args:
            batch (np.ndarray): 4-dim data minibatch
        """
        batch_transformed = []
        for x in batch:
            transformation = self.get_transformation(x.shape)
            batch_transformed.append(transformation(x))
        return np.stack(batch_transformed)


class RandomWarpAugmentation(Augmentation):
    """
    Perform random warping transformation on the input data.

    Parameters can be either constant values, a list/tuple containing the lower and upper bounds
    for a uniform distribution or a value generating function:

    Examples:
        * RandomWarpAugmentation(rotation=0.5 * np.pi)
        * RandomWarpAugmentation(rotation=(-0.25 * np.pi, 0.25 * np.pi))
        * RandomWarpAugmentation(rotation=lambda: np.random.normal(0, np.pi))

    Example Usage:

    .. code-block:: python

        aug = RandomWarpAugmentation(rotation=(0.2 * np.pi))

        # apply aug to each sample of the batch
        batch_aug = aug(batch)

        # get a transformation
        trans = aug.get_transformation(batch.shape[1:])

        # value of the rotation is available
        rot = trans.rotation

        # transform first sample in batch
        x_aug = trans(batch[0])


    Sensible starting values for parameter tuning:
        * fliph_probability = 0.5
        * flipv_probability = 0.5
        * translation = (-5, 5)
        * rotation = (-np.pi / 8, np.pi / 8)
        * scale= (0.9, 1.1)
        * shear = (-0.1 * np.pi, 0.1 * np.pi)
        * diff_scale = 8
        * diff_alpha = .75

    Args:
        fliph_probability: probability of random flips on horizontal (first) axis
        flipv_probability: probability of random flips on vertial (second) axis
        translation: translation of image data among all axis
        rotation: rotation angle in counter-clockwise direction as radians
        scale: scale as proportion of input size
        shear: shear angle in counter-clockwise direction as radians.
        diff_scale: scale parameter of diffeomorphism augmentation
        diff_alpha: intensity of diffeomorphism augmentation
        diff_fix_border: fix_border parameter of diffeomorphism augmentation
        augment_x: whether to augment input data
        augment_y: whether to augment label data
    """
    def __init__(self,
                 fliph_probability=0.,
                 flipv_probability=0.,
                 translation=(0, 0),
                 rotation=(0, 0),
                 scale=(1, 1),
                 shear=(0, 0),
                 diff_scale=1,
                 diff_alpha=0,
                 diff_fix_border=False,
                 augment_x=True,
                 augment_y=False):
        self.fliph_probability = self._parse_parameter(fliph_probability)
        self.flipv_probability = self._parse_parameter(flipv_probability)
        self.translation = self._parse_parameter(translation)
        self.rotation = self._parse_parameter(rotation)
        self.scale = self._parse_parameter(scale)
        self.shear = self._parse_parameter(shear)
        self.diff_scale = self._parse_parameter(diff_scale)
        self.diff_alpha = self._parse_parameter(diff_alpha)
        self.diff_fix_border = self._parse_parameter(diff_fix_border)

    @staticmethod
    def _parse_parameter(param):
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

    def get_transformation(self, shape):
        scale = self.scale()
        if type(self.scale()) in (float, int):
            scale = (scale, scale)

        return WarpTransformation(
            bool(random.random() < self.flipv_probability()),
            bool(random.random() < self.fliph_probability()),
            self.translation(), self.rotation(),
            scale, self.shear(), self.diff_scale(),
            self.diff_alpha(), self.diff_fix_border(),
            shape)


class WarpTransformation:
    """
    Transformation produced by ::py::class:`.RandomWarpAugmentation`.
    You can access the values of the transformation. E.g.
    WarpTransformation.translation will hold the translations of this transformation.
    """
    def __init__(self, fliph, flipv, translation, rotation, scale, shear,
                 diff_scale, diff_alpha, diff_fix_border, shape):
        self.fliph = fliph
        self.flipv = flipv
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.shear = shear
        self.diff_scale = diff_scale
        self.diff_alpha = diff_alpha
        self.diff_fix_border = diff_fix_border
        self.shape = shape[-2:]

        self.affine_transformation = self._get_affine()
        if self.diff_alpha != 0:
            self.diffeomorphism = self._get_diffeomorphism(
                self.shape, self.diff_scale, self.diff_alpha, self.diff_fix_border)
        else:
            self.diffeomorphism = None
        self.warp = self._warp_factory(self.diffeomorphism, self.affine_transformation,
                                       self.flipv, self.fliph)

    def __call__(self, img, order=3):
        return warp(img, self.warp, order=order)

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

    def _get_affine(self):
        t = AffineTransform(scale=self.scale,
                            rotation=self.rotation,
                            shear=self.shear,
                            translation=self.translation)
        return self._center_transform(t, self.shape)

    @staticmethod
    def _get_diffeomorphism(shape, scale=30, alpha=1., fix_border=True,
                            random=np.random.uniform):
        """
        Returns a diffeomorphism mapping that can be used with ``_warp_factory``.

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

    def _warp_factory(self, diff_map=None, transform=None, flipv=False, fliph=False):
        def f(xy):
            xi = xy[:, 0].astype(np.int)
            yi = xy[:, 1].astype(np.int)
            if diff_map is not None:
                off = xy + diff_map[yi, xi]
            else:
                off = xy
            if transform is not None:
                off = transform(off)
            if flipv:
                off = off[::-1, :]
            if fliph:
                off = off[:, ::-1]
            return off

        return f


def random_std(loc, scale):
    """
    Draws a random std from a gaussian distribution with mean ``loc`` and std ``scale``.
    """
    def wrapper():
        return np.clip(np.random.normal(loc=loc, scale=scale), np.finfo(float).eps, np.inf)
    return wrapper


class RandomNoiseAugmentation(Augmentation):
    """
    Add gaussian noise with variable stds.

    Args:
        std (function): Returns variable std

    """
    def __init__(self, std=random_std(0.03, 0.01)):
        self.std = std

    def get_transformation(self, shape):
        return GaussNoiseTransformation(self.std(), shape)


class GaussNoiseTransformation:
    def __init__(self, std, shape):
        self.std = std
        self.shape = shape
        self.noise = np.random.normal(loc=0., scale=self.std, size=self.shape)

    def __call__(self, arr):
        return np.clip(arr + self.noise, -1, 1)
