import random

import torch


class Compose3D:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.5), p=0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        gamma = random.uniform(*self.gamma_range)
        for key in ("fixed", "moving"):
            x = sample[key]
            min_v, max_v = x.min(), x.max()
            x = (x - min_v) / (max_v - min_v + 1e-8)
            sample[key] = x.pow(gamma) * (max_v - min_v) + min_v
        return sample


class RandomBrightness:
    def __init__(self, offset_range=(-0.1, 0.1), p=0.3):
        self.offset_range = offset_range
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        offset = random.uniform(*self.offset_range)
        sample["fixed"] = sample["fixed"] + offset
        sample["moving"] = sample["moving"] + offset
        return sample


class RandomContrast:
    def __init__(self, factor_range=(0.75, 1.25), p=0.3):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        factor = random.uniform(*self.factor_range)
        for key in ("fixed", "moving"):
            mean = sample[key].mean()
            sample[key] = (sample[key] - mean) * factor + mean
        return sample


class RandomGaussianNoise:
    def __init__(self, std_range=(0.0, 0.03), p=0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        std = random.uniform(*self.std_range)
        sample["fixed"] = sample["fixed"] + torch.randn_like(sample["fixed"]) * std
        sample["moving"] = sample["moving"] + torch.randn_like(sample["moving"]) * std
        return sample


class RandomFlip3D:
    def __init__(self, axes=(1, 2, 3), p=0.0):
        self.axes = axes
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        dims = [axis for axis in self.axes if random.random() < 0.5]
        if not dims:
            return sample
        for key, value in list(sample.items()):
            if torch.is_tensor(value) and value.dim() >= 4 and key in {"fixed", "moving", "fixed_seg", "moving_seg", "fixed_mask", "moving_mask"}:
                sample[key] = torch.flip(value, dims=dims)
        return sample


class RegistrationAugmentation:
    def __init__(self, enable=False, gamma=True, brightness=True, contrast=True, noise=True, flip=False):
        transforms = []
        if enable:
            if gamma:
                transforms.append(RandomGamma())
            if brightness:
                transforms.append(RandomBrightness())
            if contrast:
                transforms.append(RandomContrast())
            if noise:
                transforms.append(RandomGaussianNoise())
            if flip:
                transforms.append(RandomFlip3D(p=0.5))
        self.transform = Compose3D(transforms)

    def __call__(self, sample):
        return self.transform(sample)
