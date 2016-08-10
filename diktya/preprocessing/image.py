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
    def __init__(self,
                 flipx=lambda: random.getrandbits(1),
                 flipy=lambda: random.getrandbits(1),
                 translation=lambda: np.random.randint(-5, 5),
                 rotation=lambda: np.random.uniform(-np.pi / 8, np.pi / 8),
                 scale=lambda: np.random.uniform(0.9, 1.1),
                 shear=lambda: np.random.uniform(-0.1 * np.pi, 0.1 * np.pi),
                 use_diff=lambda: True,
                 diff_scale=lambda: 8,
                 diff_alpha=lambda: .75,
                 diff_fix_border=lambda: False,
                 augment_x=True,
                 augment_y=False):
        self.flipx = flipx
        self.flipy = flipy
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.shear = shear
        self.use_diff = use_diff
        self.diff_scale = diff_scale
        self.diff_alpha = diff_alpha
        self.diff_fix_border = diff_fix_border
        self.augment_x = augment_x
        self.augment_y = augment_y

    @staticmethod
    def center_transform(transform, shape):
        center_transform = AffineTransform(translation=(-shape[0] // 2, -shape[1] // 2))
        uncenter_transform = AffineTransform(translation=(shape[0] // 2, shape[1] // 2))
        return center_transform + transform + uncenter_transform

    @staticmethod
    def get_frame(shape, line_width, blur=8):
        frame = np.zeros(shape)
        frame[:, :line_width] = 1
        frame[:, -line_width:] = 1
        frame[:line_width, :] = 1
        frame[-line_width:, :] = 1
        return frame

    def get_random_affine(self, shape):
        t = AffineTransform(scale=(self.scale(), self.scale()),
                            rotation=self.rotation(),
                            shear=self.shear(),
                            translation=self.translation())
        return RandomWarpAugmentation.center_transform(t, shape)


    def get_diffeomorphism(self, shape, scale=30, alpha=1., fix_border=True, random=np.random.uniform):
        """
        Returns a diffeomorphism mapping that can be used wtih ``diff_warp``.

        Args:
            shape (tuple): Shape of the image
            scale: Scale of the diffeomorphism in pixels.
            alpha (float): Intensity of the diffeomorphism. Must be between 0 and 1
            fix_border (boolean): If true the border of the resulting image stay constant.
            random: Function to draw the randomness.  Will be called with ``random(-intensity, intensity, shape)``.
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
            frame = RandomWarpAugmentation.get_frame((dh, dw), 1)
            for i in (0, 1):
                diff_map[:, :, i] = diff_map[:, :, i] * (1 - frame)

        diff_map = resize(diff_map, (h, w, 2), order=3)
        return diff_map

    def get_random_diffeomorphism(self, shape):
        return self.get_diffeomorphism(shape,
                                       scale=self.diff_scale(),
                                       alpha=self.diff_alpha(),
                                       fix_border=self.diff_fix_border())

    def diff_warp(self, diff_map, transform=None, flipx=False, flipy=False):
        def f(xy):
            xi = xy[:, 0].astype(np.int)
            yi = xy[:, 1].astype(np.int)
            off = xy + diff_map[yi, xi]
            if transform is not None:
                off = transform(off)
            if flipx:
                off = off[::-1, :]
            if flipy:
                off = off[:, ::-1]
            return off

        return f

    def __call__(self, x, y):
        if self.augment_x and self.augment_y:
            assert (x.shape == y.shape)

        transform = self.get_random_affine(x.shape)
        diff = self.get_random_diffeomorphism(x.shape)
        warp_f = self.diff_warp(diff, transform, self.flipx(), self.flipy())

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