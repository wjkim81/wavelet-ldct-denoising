import random
import numpy as np
import skimage.color as sc

import torch

def add_noise(x, noise=0):
    if noise == 0:
        noise_value = np.random.randint(55)
    else:
        noise_value = noise

    noises = np.random.normal(scale=noise_value, size=x.shape)
    noises = noises.round()
        
    x_noise = x.astype(np.int16) + noises.astype(np.int16)
    x_noise = x_noise.clip(0, 255).astype(np.uint8)
    return x_noise

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if img.ndim == 2:
            if hflip: img = img[:, ::-1].copy()
            if vflip: img = img[::-1, :].copy()
            if rot90: img = img.transpose(1, 0).copy()
        elif img.ndim == 3:
            if hflip: img = img[:, ::-1, :].copy()
            if vflip: img = img[::-1, :, :].copy()
            if rot90: img = img.transpose(1, 0, 2).copy()
            
        return img

    return [_augment(a) for a in args]

def get_patch(*args, patch_size=96, n_channels=1, scale=1, multi=False, input_large=False):

    ih, iw = args[0].shape[:2]

    tp = patch_size
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    if n_channels == 1:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]
    else:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1.0 / rgb_range)

        return tensor

    return [_np2Tensor(a) for a in args]


def get_3dpatch(*args, patch_size=96, depth=3, n_channels=1):

    idp, ih, iw = args[0].shape[:3]

    pd = depth

    tp = patch_size
    ip = patch_size

    iz = random.randrange(0, idp - pd + 1)
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    tz, tx, ty = iz, ix, iy

    if n_channels == 1:
        ret = [
            args[0][iz:iz + pd, iy:iy + ip, ix:ix + ip],
            *[a[tz:tz + pd, ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]
    else:
        ret = [
            args[0][iz:iz + pd, iy:iy + ip, ix:ix + ip, :],
            *[a[tz:tz + pd, ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]

    return ret

def set_channel3d(*args):
    def _set_channel(img):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=3)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor3d(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((3, 0, 1, 2)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1.0 / rgb_range)

        return tensor

    return [_np2Tensor(a) for a in args]


def augment3d(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if img.ndim == 3:
            if hflip: img = img[:, :, ::-1].copy()
            if vflip: img = img[:, ::-1, :].copy()
            if rot90: img = img.transpose(0, 2, 1).copy()
        elif img.ndim == 4:
            if hflip: img = img[:, :, ::-1, :].copy()
            if vflip: img = img[:, ::-1, :, :].copy()
            if rot90: img = img.transpose(0, 2, 1, 3).copy()
            
        return img

    return [_augment(a) for a in args]