from collections import Iterable
from numbers import Number
from types import FunctionType
import random

import numpy as np
from skimage.transform import warp, resize, AffineTransform
from skimage.exposure import equalize_hist


def chain_augmentations(*augmentations, augment_x=True, augment_y=False):
    """
    Chain multiple augmentations.

    Example:

    .. code-block:: python

        aug = chain_augmentations(NoiseAugmentation(),
                                  WarpAugmentation())

        Xa, ya = aug((X, y))

    Args:
        augmentations (Augmentation or functions):
            Can be a Augmentation object or any callable object. The leftest
            augmentations is applied first.
        augment_x (bool): should augment data X
        augment_y (bool): should augment label y

    Returns:
        A function that takes a minibatch as input and applies the augmentations.
        The minibatch can either be a numpy array or tuple of (X, y)
        where X are the data and y the labels.

    """
    def get_transformation(aug, shape):
        if issubclass(type(aug), Augmentation):
            return aug.get_transformation(shape)
        elif hasattr(aug, '__call__'):
            return aug
        else:
            raise Exception("Must be a subclass of Augmentation or callabe. "
                            "But got {}".format(aug))

    def wrapper(batch):
        if type(batch) == tuple:
            X, Y = batch
            if len(X) != len(Y):
                raise Exception("Got tuple but arrays have different size. "
                                "Got X= {} and y={}".format(X.shape, Y.shape))
            Y_aug = []
        elif type(batch) == np.ndarray:
            X = batch
            Y = None
            Y_aug = None
        X_aug = []
        for i in range(len(X)):
            x = X[i]
            if Y is not None:
                y = Y[i]
            else:
                y = None
            for aug in augmentations:
                transformation = get_transformation(aug, X[i].shape)
                if augment_x:
                    x = transformation(x)
                if augment_y and y is not None:
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
        to the augmented image. See :py:meth:`.WarpAugmentation.transformation`
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


class CropAugmentation(Augmentation):
    def __init__(self, translation, crop_shape):
        self.translation = _parse_parameter(translation)
        self.crop_shape = crop_shape

    def get_transformation(self, shape):
        return CropTransformation([int(self.translation()), int(self.translation())],
                                  self.crop_shape)


class CropTransformation:
    def __init__(self, translation, crop_shape):
        if type(translation) == int:
            translation = (translation, translation)
        if type(translation[0]) != int or type(translation[1]) != int:
            raise Exception("Translation must be an integer! But got {}".format(translation))
        self.translation = translation
        self.crop_shape = crop_shape

    def __call__(self, data):
        if len(data.shape) <= 1:
            raise Exception("Shape must be at least 2-dimensional. Got {}."
                            .format(data.shape))
        crop_shp = self.crop_shape
        if data.shape[-2:] != crop_shp:
            h, w = data.shape[-2:]
            assert h >= crop_shp[0] and w >= crop_shp[1]
            hc = h // 2 + self.translation[0]
            wc = w // 2 + self.translation[1]
            hb = max(hc - crop_shp[0] // 2, 0)
            he = hb + crop_shp[0]
            wb = max(wc - crop_shp[1] // 2, 0)
            we = wb + crop_shp[1]
            return data[..., hb:he, wb:we]
        else:
            return data


class HistEqualization:
    """
    Performs historgram equalization. See ``skimage.expose.equalize_hist``.
    The returned data is scaled to ``[-1, 1]``.
    """
    def __call__(self, data):
        return 2*equalize_hist(data) - 1


class ChannelScaleShiftAugmentation(Augmentation):
    def __init__(self, scale, shift, min=-1, max=1, per_channel=True):
        """
        Augments a image by scaling and shifts its channels.
        """
        self.scale = _parse_parameter(scale)
        self.shift = _parse_parameter(shift)
        self.min = min
        self.max = max
        self.per_channel = per_channel

    def get_transformation(self, shape):
        if len(shape) != 3:
            raise Exception("Shape must be 3-dimensional. Got {}.".format(shape))
        nb_channels = shape[0]
        if self.per_channel:
            shift = [self.shift() for _ in range(nb_channels)]
            scale = [self.scale() for _ in range(nb_channels)]
        else:
            shift = [self.shift()] * nb_channels
            scale = [self.scale()] * nb_channels
        return ChannelScaleShiftTransformation(scale, shift, self.min, self.max)


class ChannelScaleShiftTransformation():
    def __init__(self, scale, shift, min=-1, max=1):
        self.scale = scale
        self.shift = shift
        self.min = min
        self.max = max

    def __call__(self, x):
        return np.stack(
            [np.clip(self.scale[i]*channel + self.shift[i], self.min, self.max)
             for i, channel in enumerate(x)])


class WarpAugmentation(Augmentation):
    """
    Perform random warping transformation on the input data.

    Parameters can be either constant values, a list/tuple containing the lower and upper bounds
    for a uniform distribution or a value generating function:

    Examples:
        * WarpAugmentation(rotation=0.5 * np.pi)
        * WarpAugmentation(rotation=(-0.25 * np.pi, 0.25 * np.pi))
        * WarpAugmentation(rotation=lambda: np.random.normal(0, np.pi))

    Example Usage:

    .. code-block:: python

        aug = WarpAugmentation(rotation=(0.2 * np.pi))

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
        * diffeomorphism = [(8, .75)]

    Args:
        fliph_probability: probability of random flips on horizontal (first) axis
        flipv_probability: probability of random flips on vertial (second) axis
        translation: translation of image data among all axis
        rotation: rotation angle in counter-clockwise direction as radians
        scale: scale as proportion of input size
        shear: shear angle in counter-clockwise direction as radians.
        diffeomorphism: list of diffeomorphism parameters. Elements must
            be of ``(scale, intensity)``.
        diff_fix_border: fix_border parameter of diffeomorphism augmentation
        fill_mode (default 'edge'): one of corresponse to numpy.pad mode

    """
    def __init__(self,
                 fliph_probability=0.,
                 flipv_probability=0.,
                 translation=(0, 0),
                 rotation=(0, 0),
                 scale=(1, 1),
                 shear=(0, 0),
                 diffeomorphism=[],
                 diff_fix_border=False,
                 fill_mode='edge',
                 ):
        self.fliph_probability = _parse_parameter(fliph_probability)
        self.flipv_probability = _parse_parameter(flipv_probability)
        self.translation = _parse_parameter(translation)
        self.rotation = _parse_parameter(rotation)
        self.scale = _parse_parameter(scale)
        self.shear = _parse_parameter(shear)
        self.diffeomorphism = [(_parse_parameter(s), _parse_parameter(i))
                               for s, i in diffeomorphism]
        self.diff_fix_border = _parse_parameter(diff_fix_border)
        self.fill_mode = fill_mode

    def get_transformation(self, shape):
        scale = self.scale()
        if type(self.scale()) in (float, int):
            scale = (scale, scale)

        diffeomorphism = [(diff_scale(), diff_intensity())
                          for diff_scale, diff_intensity in self.diffeomorphism]
        return WarpTransformation(
            bool(random.random() < self.flipv_probability()),
            bool(random.random() < self.fliph_probability()),
            (self.translation(), self.translation()),
            self.rotation(),
            scale, self.shear(),
            diffeomorphism, self.diff_fix_border(),
            self.fill_mode,
            shape)


class WarpTransformation:
    """
    Transformation produced by ::py::class:`.WarpAugmentation`.
    You can access the values of the transformation. E.g.
    WarpTransformation.translation will hold the translations of this transformation.
    """
    def __init__(self, fliph, flipv, translation, rotation, scale, shear,
                 diffeomorphism, diff_fix_border, fill_mode, shape):
        self.fliph = fliph
        self.flipv = flipv
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.shear = shear
        self.diffeomorphism = diffeomorphism
        self.diff_fix_border = diff_fix_border
        self.fill_mode = fill_mode
        self.shape = shape[-2:]

        self.affine_transformation = self._get_affine()
        if self.diffeomorphism:
            self.diffeomorphism_map = sum([
                self._get_diffeomorphism_map(
                    self.shape, diff_scale, diff_intensity, self.diff_fix_border)
                for diff_scale, diff_intensity in self.diffeomorphism])

        else:
            self.diffeomorphism_map = None
        self.warp = self._warp_factory(self.diffeomorphism_map, self.affine_transformation,
                                       self.flipv, self.fliph)

    def __call__(self, img, order=3):
        if img.ndim == 3:
            img_warped = []
            for channel in img:
                img_warped.append(warp(channel, self.warp, order=order, mode=self.fill_mode))
            return np.stack(img_warped)
        elif img.ndim == 2:
            return warp(img, self.warp, order=order, mode=self.fill_mode)
        else:
            raise Exception("Wrong number of dimensions. Expected 2 or 3. "
                            "Got {} with shape {}.".format(img.ndim, img.shape))

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
    def _get_diffeomorphism_map(shape, scale=30, intensity=1., fix_border=True,
                                random=np.random.uniform):
        """
        Returns a diffeomorphism mapping that can be used with ``_warp_factory``.

        Args:
            shape (tuple): Shape of the image
            scale: Scale of the diffeomorphism in pixels.
            intensity (float): Intensity of the diffeomorphism. Must be between 0 and 1
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
        intensity = 0.25 * intensity * 1 / rel_scale
        diff_map = np.clip(random(-intensity, intensity, (dh, dw, 2)), -intensity, intensity)
        if fix_border:
            frame = WarpTransformation._get_frame((dh, dw), 1)
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


class NoiseAugmentation(Augmentation):
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


class LambdaAugmentation(Augmentation):
    def __init__(self, func, **params):
        self.func = func
        self.params = {k: _parse_parameter(v) for k, v in params.items()}

    def get_transformation(self, shape):
        transformation_params = {k: param() for k, param in self.params.items()}
        return LambdaTransformation(self.func, transformation_params)


class LambdaTransformation:
    def __init__(self, func, params):
        self.func = func
        self.params = params

    def __call__(self, arr):
        return self.func(arr, **self.params)
