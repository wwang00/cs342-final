# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, image2, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            image2 = F.hflip(image2)
            args = tuple(np.array([(image.width-x1, y0, image.width-x0, y1) for x0, y0, x1, y1 in boxes])
                         for boxes in args)
        return (image,image2) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, image2, *args):
        for t in self.transforms:
            image, image2, *args = t(image, image2, *args)
        return (image, image2) + tuple(args)


class Normalize(T.Normalize):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, image2, *args):
        return (F.to_tensor(image),F.to_tensor(image2)) + args


class ToHeatmap(object):
    def __init__(self, radius=8):
        self.radius = radius

    def __call__(self, image, image2, *dets):
        peak, size = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)
        return image, image2, peak, size, len(dets[1]) > 0


# TODO try increasing radius?
def detections_to_heatmap(dets, shape, radius=2, device=None):
    radius = 8
    with torch.no_grad():
        size = torch.zeros((2, shape[0], shape[1]), device=device)
        peak = torch.zeros((len(dets), shape[0], shape[1]), device=device)
        for i, det in enumerate(dets):
            if len(det):
                det = torch.tensor(det.astype(float), dtype=torch.float32, device=device)
                cx, cy = (det[:, 0] + det[:, 2] - 1) / 2, (det[:, 1] + det[:, 3] - 1) / 2
                x = torch.arange(shape[1], dtype=cx.dtype, device=cx.device)
                y = torch.arange(shape[0], dtype=cy.dtype, device=cy.device)
                gx = (-((x[:, None] - cx[None, :]) / radius)**2).exp()
                gy = (-((y[:, None] - cy[None, :]) / radius)**2).exp()
                gaussian, id = (gx[None] * gy[:, None]).max(dim=-1)
                mask = gaussian > peak.max(dim=0)[0]
                det_size = (det[:, 2:] - det[:, :2]).T / 2
                size[:, mask] = det_size[:, id[mask]]
                peak[i] = gaussian
        return peak, size
