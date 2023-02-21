import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import PIL
from PIL import Image
from util.misc import interpolate
from skimage import transform


def hflip(image, target):
    # flip
    flipped_image = image
    if random.random() < 0.5:
        flipped_image = np.flip(flipped_image, (1))
        target = target.copy()
        if "masks" in target:
            mask = target['masks']
            target['masks'] = np.flip(mask, (1))
    if random.random() < 0.5:
        flipped_image = np.flip(flipped_image, (2))
        target = target.copy()
        if "masks" in target:
            mask = target['masks']
            target['masks'] = np.flip(mask, (2))
    if random.random() < 0.5:
        flipped_image = np.flip(flipped_image, (3))
        target = target.copy()
        if "masks" in target:
            mask = target['masks']
            target['masks'] = np.flip(mask, (3))
            target['masks'] = np.flip(mask, (3))
    # rotation
    rotate_choice = int(random.random() * 4)
    flipped_image = np.rot90(flipped_image, k=rotate_choice, axes=(1, 2))
    if "masks" in target:
        mask = target['masks']
        target['masks'] = np.rot90(mask, k=rotate_choice, axes=(1, 2))
    return flipped_image, target


def crop_3D(image, target, region):
    i, j, t, h, w, l = region
    cropped_image = image[:, i:i + h, j:j + w, t:t + l]
    target["size"] = [h, w, l]

    if "masks" in target:
        mask = target["masks"]
        target['masks'] = mask[:, i:i + h, j:j + w, t:t + l]
    return cropped_image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        _, dx, dy, dz = img.shape
        target_x, target_y, target_z = self.size
        if target_x < dx:
            crop_x = int(round((dx - target_x) / 2.))
        else:
            crop_x = 0
        if target_y < dy:
            crop_y = int(round((dy - target_y) / 2.))
        else:
            crop_y = 0
        if target_z < dz:
            crop_z = int(round((dz - target_z) / 2.))
        else:
            crop_z = 0
        return crop_3D(img, target, (crop_x, crop_y, crop_z, target_x, target_y, target_z))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        return hflip(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        for k, v in target.items():
            if not isinstance(v, str):
                if torch.is_tensor(v) or isinstance(v, (list, tuple)):
                    if torch.is_tensor(v):
                        pass
                    else:
                        target[k] = torch.tensor(np.float32(v)).type(torch.LongTensor)
                else:
                    v = v.copy()
                    target[k] = torch.tensor(np.float32(v)).type(torch.LongTensor)
        if not torch.is_tensor(img):
            img = img.copy()
            img = torch.from_numpy(img)
        return img, target
        # return torch.from_numpy(img), target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if self.mean is None:
            self.mean = image.mean()
        if self.std is None:
            self.std = image.std()
        image = (image - self.mean) / self.std
        if target is None:
            return image, None
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
