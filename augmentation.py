# -*- coding: utf-8 -*-

from enum import IntEnum

import cv2 as cv
import numpy as np
from albumentations import RandomSizedCrop, RandomSunFlare


class FlipDirection(IntEnum):
    BOTH = -1
    HORIZONTAL = 0
    VERTICAL = 1


def rotate_with_borders(rot_im, angle):
    rot_h, rot_w = rot_im.shape[:2]
    center_x, center_y = rot_w / 2, rot_h / 2
    rotate_m = cv.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    cos = np.abs(rotate_m[0, 0])
    sin = np.abs(rotate_m[0, 1])

    target_w = int((rot_h * sin) + (rot_w * cos))
    target_h = int((rot_h * cos) + (rot_w * sin))

    rotate_m[0, 2] += (target_w / 2) - center_x
    rotate_m[1, 2] += (target_h / 2) - center_y

    return cv.warpAffine(rot_im, rotate_m, (target_w, target_h))


class MotionBlurAug(object):

    def __init__(self, p=1.0, kernel_sizes=range(3, 10, 2), directions=range(4)):
        self.aug_prob = p
        self.directions = directions
        self.kernel_sizes = kernel_sizes

    @staticmethod
    def motion_blur_image(im, kernel_size=9, direction=0):
        if direction == 0:
            # Horizontal
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
        elif direction == 1:
            # Vertical
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
            kernel = kernel / kernel_size
        elif direction == 2:
            # Diagonal
            kernel = np.zeros((kernel_size, kernel_size))
            np.fill_diagonal(kernel, 1)
            kernel = kernel / kernel_size
        else:
            # Anti-Diagonal
            kernel = np.zeros((kernel_size, kernel_size))
            np.fill_diagonal(np.fliplr(kernel), 1)
            kernel = kernel / kernel_size

        output = cv.filter2D(im, -1, kernel)
        return output

    def augment(self, im, mask):
        if np.random.rand() > self.aug_prob:
            return im, mask

        k = np.random.choice(self.kernel_sizes)
        d = np.random.choice(self.directions)

        aug = self.motion_blur_image(im, kernel_size=k, direction=d)
        return aug, mask


class GaussianBlurAug(object):

    def __init__(self, p=1.0, kernel_sizes=range(3, 6, 2)):
        self.aug_prob = p
        self.kernel_sizes = kernel_sizes

    def augment(self, im, mask):
        if np.random.rand() > self.aug_prob:
            return im, mask

        k = np.random.choice(self.kernel_sizes)

        aug = cv.GaussianBlur(im, (k, k), 0)
        return aug, mask


class ResizeAug(object):

    def __init__(self, p=1.0, resize_min=0.1, resize_max=0.5):
        self.aug_prob = p
        self.resize_min = resize_min
        self.resize_max = resize_max

    def augment(self, im, mask):
        if np.random.rand() > self.aug_prob:
            return im, mask

        factor = np.random.uniform(self.resize_min, self.resize_max)
        new_im = cv.resize(im, None, fx=factor, fy=factor)
        new_im = cv.resize(new_im, im.shape[::-1][-2:])
        return new_im, mask


class MedianBlurAug(object):

    def __init__(self, p=1.0, kernel_sizes=range(3, 20, 2)):
        self.aug_prob = p
        self.kernel_sizes = kernel_sizes

    def augment(self, im, mask):
        if np.random.rand() > self.aug_prob:
            return im, mask

        k = np.random.choice(self.kernel_sizes)

        aug = cv.medianBlur(im, k)
        return aug, mask


class ColorJitterAug(object):

    def __init__(self, p=1.0, saturation_min=0.9, saturation_max=1.0, hue_min=0.9, hue_max=1.1, value_min=0.9,
                 value_max=1.1):
        self.aug_prob = p
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.value_min = value_min
        self.value_max = value_max

    def augment(self, im, mask):
        if np.random.rand() > self.aug_prob:
            return im, mask

        im_out = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        saturation_factor = self.saturation_min + np.random.rand() * (self.saturation_max - self.saturation_min)
        hue_factor = self.hue_min + np.random.rand() * (self.hue_max - self.hue_min)
        value_factor = self.value_min + np.random.rand() * (self.value_max - self.value_min)

        im_out[:, :, 0] = np.clip((im_out[:, :, 0] * hue_factor), 0, 179).astype(np.uint8)  # H is in [0, 179] in OpenCV
        im_out[:, :, 1] = np.clip((im_out[:, :, 1] * saturation_factor), 0, 255).astype(np.uint8)
        im_out[:, :, 2] = np.clip((im_out[:, :, 2] * value_factor), 0, 255).astype(np.uint8)

        im_out = cv.cvtColor(im_out, cv.COLOR_HSV2BGR)

        if im.shape[2] == 4:
            # Original image has alpha channel, so concatenate it back
            im_out = np.dstack((im_out[:, :, 0], im_out[:, :, 1], im_out[:, :, 2], im[:, :, 3]))

        return im_out, mask


class GlareAugmentation(object):

    def __init__(self, p=1.0, glare_min_count=1, glare_max_count=1, glare_x_min=-0.1, glare_x_max=0.3, glare_y_min=-0.1,
                 glare_y_max=0.3, glare_w_min=0.17, glare_w_max=0.8, glare_h_min=0.08, glare_h_max=0.3,
                 glare_color=(237, 248, 252), glare_angle_max=180, glare_shade_dist_min=0.0, glare_shade_dist_max=0.4):
        self.p = p
        self.glare_min_count = glare_min_count
        self.glare_max_count = glare_max_count
        self.glare_w_min = glare_w_min
        self.glare_w_max = glare_w_max
        self.glare_h_min = glare_h_min
        self.glare_h_max = glare_h_max
        self.glare_color = glare_color
        self.glare_angle_max = glare_angle_max
        self.glare_x_min = glare_x_min
        self.glare_x_max = glare_x_max
        self.glare_y_min = glare_y_min
        self.glare_y_max = glare_y_max
        self.glare_shade_dist_min = glare_shade_dist_min
        self.glare_shade_dist_max = glare_shade_dist_max

    @staticmethod
    def alpha_ellipse(half_w, half_h, shade_dist=0.0):
        if shade_dist < 0.0 or shade_dist > 1.0:
            raise RuntimeError("Invalid Value for shade_dist")

        result = np.zeros((2 * half_h, 2 * half_w))
        for r_idx in range(2 * half_h):
            for c_idx in range(2 * half_w):
                x = half_w - c_idx
                y = half_h - r_idx

                ellipse_ratio = x ** 2 / half_w ** 2 + y ** 2 / half_h ** 2
                if ellipse_ratio >= 1.0:
                    score = 0.0
                elif ellipse_ratio <= shade_dist:
                    score = 1.0
                else:
                    score = 1.0 - ((ellipse_ratio - shade_dist) / (1.0 - shade_dist))
                result[r_idx, c_idx] = score
        return result

    def random_glare_on_image(self, bg_im):
        im_h, im_w = bg_im.shape[:2]
        im_d = np.sqrt(im_h * im_w)
        glare_w = int(im_d * np.random.uniform(self.glare_w_min, self.glare_w_max))
        glare_h = int(im_d * np.random.uniform(self.glare_h_min, self.glare_h_max))
        glare_x = int(im_d * np.random.uniform(self.glare_x_min, self.glare_x_max))
        glare_y = int(im_d * np.random.uniform(self.glare_y_min, self.glare_y_max))
        glare_angle = np.random.randint(self.glare_angle_max)
        shade_dist = np.random.uniform(self.glare_shade_dist_min, self.glare_shade_dist_max)

        alpha = self.alpha_ellipse(glare_w, glare_h, shade_dist=shade_dist)
        alpha = rotate_with_borders(alpha, glare_angle)[:, :, np.newaxis]

        paint_h, paint_w = alpha.shape[:2]
        output_im = bg_im.copy()
        output_h, output_w = bg_im.shape[:2]

        target_x1 = max(glare_x, 0)
        target_y1 = max(glare_y, 0)
        target_x2 = min(max(glare_x + paint_w, 0), output_w)
        target_y2 = min(max(glare_y + paint_h, 0), output_h)
        draw_h = target_y2 - target_y1
        draw_w = target_x2 - target_x1

        if draw_h <= 0 or draw_w <= 0:
            return bg_im

        glare_im = np.zeros((paint_h, paint_w, 3), dtype=np.uint8)
        glare_im[:, :, :] = self.glare_color

        bg_slice = bg_im[target_y1:target_y2, target_x1: target_x2]
        alpha_slice = alpha[:draw_h, :draw_w]
        glare_slice = glare_im[:draw_h, :draw_w]

        def add_weighted(first, second, alp):
            return (alp * first + (1 - alp) * second).astype(np.uint8)

        output_im[target_y1:target_y2, target_x1:target_x2] = add_weighted(glare_slice, bg_slice, alpha_slice)
        return output_im

    def augment(self, im, mask):
        if np.random.rand() > self.p:
            return im, mask

        num_glares = np.random.randint(self.glare_min_count, self.glare_max_count + 1)
        aug_im = im.copy()[:, :, :3]

        for _ in range(num_glares):
            aug_im = self.random_glare_on_image(aug_im)

        if im.shape[2] == 4:
            # Original image has alpha channel, so concatenate it back
            aug_im = np.dstack((aug_im[:, :, 0], aug_im[:, :, 1], aug_im[:, :, 2], im[:, :, 3]))

        return aug_im, mask


class FlipAug(object):

    def __init__(self, p=1.0, directions=(FlipDirection.BOTH, FlipDirection.HORIZONTAL, FlipDirection.VERTICAL)):
        self.aug_prob = p
        self.directions = directions

    def augment(self, im, mask):
        if np.random.rand() > self.aug_prob:
            return im, mask

        d = np.random.choice(self.directions)
        im_flipped = cv.flip(im, d)
        mask_flipped = cv.flip(mask, d)

        return im_flipped, mask_flipped


class PickOne(object):

    def __init__(self, augmentations, probs=None):
        self.augmentations = augmentations
        self.probs = probs

    def augment(self, im, mask):
        aug = np.random.choice(self.augmentations, p=self.probs)
        return aug.augment(im, mask)


class ApplyAll(object):

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def augment(self, im_x, im_y):
        im_x_out = im_x.copy()
        im_y_out = im_y.copy()

        for augmentation in self.augmentations:
            im_x_out, im_y_out = augmentation.augment(im_x_out, im_y_out)

        return im_x_out, im_y_out


class AlbuAug(object):

    def __init__(self, aug):
        """
        Augmentation using the albumentations library
        :param aug: A DualTransform augmentation which will be applied both to image and to mask
        """
        self.aug = aug

    def augment(self, im, mask):
        augmented = self.aug(image=im, mask=mask)

        im_aug = augmented["image"]
        mask_aug = augmented["mask"]

        return im_aug, mask_aug


class RandomSizedCropAlbuAug(AlbuAug):

    def __init__(self, prob, size_factor=0.25):
        aug = RandomSizedCrop(p=prob, min_max_height=(1, 1), height=1, width=1)
        super().__init__(aug)
        self.size_factor = size_factor

    def augment(self, im, mask):
        h, w = im.shape[:2]
        self.aug.min_max_height = (int(self.size_factor * h), h)
        self.aug.height = h
        self.aug.width = w
        self.aug.w2h_ratio = w / h
        return super().augment(im, mask)


class FlareAlbuAug(AlbuAug):

    def __init__(self, aug):
        super().__init__(aug)
        if not isinstance(self.aug, RandomSunFlare):
            raise RuntimeError("Expected RandomSunFlare Augmentation")

    def augment(self, im, mask):
        h, w, c = im.shape

        if c == 4:
            no_alpha = im[:, :, :3]
            alpha = im[:, :, 3]
            i, m = super().augment(no_alpha, mask)[0]
            im_out = np.dstack((i[:, :, 0], i[:, :, 1], i[:, :, 2], alpha))
            return im_out, m
        return super().augment(im, mask)
