import json
from enum import unique, Enum

import cv2 as cv
import numpy as np
import torch


CHECKPOINT_MODEL = "model"
CHECKPOINT_OPTIMIZER = "optimizer"
CHECKPOINT_EPOCH_ID = "epoch"
CHECKPOINT_ADDITIONAL_DATA = "additional_data"


def load_json(path, *args, **kwargs):
    with open(path, "r") as f:
        return json.load(f, *args, **kwargs)


def save_json(path, data, prettify=True, *args, **kwargs):
    with open(path, "w") as f:
        if prettify:
            json.dump(data, f, indent=4, sort_keys=True, separators=(",", ": "), ensure_ascii=False, *args, **kwargs)
        else:
            json.dump(data, f, *args, **kwargs)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state[CHECKPOINT_MODEL])
    if optimizer is not None:
        optimizer.load_state_dict(state[CHECKPOINT_OPTIMIZER])
    epoch = state.get(CHECKPOINT_EPOCH_ID)
    additional_data = state.get(CHECKPOINT_ADDITIONAL_DATA)
    return epoch, additional_data


def save_checkpoint(path, model, optimizer=None, epoch=None, additional_data=None):
    torch.save({
        CHECKPOINT_MODEL: model.state_dict(),
        CHECKPOINT_OPTIMIZER: optimizer.state_dict() if optimizer is not None else None,
        CHECKPOINT_EPOCH_ID: epoch,
        CHECKPOINT_ADDITIONAL_DATA: additional_data
    }, path)


def rgb_bgr(x):
    return x[..., ::-1].copy()


def opencv_show(*ims, prefix="", titles=None):
    for idx, im in enumerate(ims):
        title = f"{prefix}{idx}" if titles is None else titles[idx]
        cv.imshow(title, im)
    cv.waitKey()
    cv.destroyAllWindows()


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w


def put_side_by_side(images):
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]

    images = [cv.cvtColor(el, cv.COLOR_GRAY2BGR) if len(el.shape) == 2 else el for el in images]
    first = images[0]

    h, w, c = first.shape
    sbs = np.zeros((h, len(images) * w, c), dtype=np.uint8)
    sbs[:, :w, :] = first
    for idx, other_img in enumerate(images[1:]):
        sbs[:, (idx + 1) * w:(idx + 2) * w, :] = other_img

    return sbs


@unique
class NoiseType(Enum):
    GAUSSIAN = "GAUSSIAN"
    SALT_AND_PEPPER = "SALT_AND_PEPPER"
    POISSON = "POISSON"
    SPECKLE = "SPECKLE"


def add_gaussian_noise(im, mean=0, std=25):
    gauss = np.random.normal(mean, std, im.shape)
    return np.clip(im + gauss, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(im, salt_ratio=0.5, amount=0.008):
    im_noise = im.copy()
    coords = tuple([np.random.randint(0, i - 1, int(amount * im.size * salt_ratio))
                    for i in im.shape[:2]])
    im_noise[coords] = 255

    coords = tuple([np.random.randint(0, i - 1, int(amount * im.size * (1.0 - salt_ratio)))
                    for i in im.shape[:2]])
    im_noise[coords] = 0
    return im_noise


def add_poisson_noise(im, noise_strength=None):
    if noise_strength is None:
        unique_vals = len(np.unique(im))
        noise_strength = unique_vals
    return np.clip(np.random.poisson(im / 255 * noise_strength) / noise_strength * 255, 0, 255).astype(np.uint8)


def add_speckle_noise(im, factor=0.5):
    gauss = np.random.randn(*im.shape)
    return np.clip(im + factor * im * gauss, 0, 255).astype(np.uint8)


def add_noise(noise_type, im):
    if noise_type == NoiseType.GAUSSIAN:
        return add_gaussian_noise(im)
    elif noise_type == NoiseType.SALT_AND_PEPPER:
        return add_salt_and_pepper_noise(im)
    elif noise_type == NoiseType.POISSON:
        return add_poisson_noise(im)
    elif noise_type == NoiseType.SPECKLE:
        return add_speckle_noise(im)