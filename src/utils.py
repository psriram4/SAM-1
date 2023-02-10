from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101

from data.tranforms import TransformTrain
from data.tranforms import TransformTest
import data
from torch.utils.data import DataLoader, RandomSampler
import os

import matplotlib.pyplot as plt
import random
import colorsys
from skimage.measure import find_contours
import cv2
import numpy as np
from matplotlib.patches import Polygon

imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)

def load_data(args):
    batch_size_dict = {"train": args.batch_size, "test": 100}

    transform_train = TransformTrain()
    transform_test = TransformTest(mean=imagenet_mean, std=imagenet_std)
    dataset = data.__dict__[os.path.basename(args.root)]

    datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio, download=True, transform=transform_train)}
    # datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio, download=True)}
    test_dataset = {
        'test' + str(i): dataset(root=args.root, split='test', label_ratio=100, download=True, transform=transform_test["test" + str(i)]) for i in range(10)
    }
    datasets.update(test_dataset)

    dataset_loaders = {x: DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True, num_workers=4)
                        for x in ['train']}
    dataset_loaders.update({'test' + str(i): DataLoader(datasets["test" + str(i)], batch_size=4, shuffle=False, num_workers=4)
                            for i in range(10)})

    return dataset_loaders



def load_network(backbone):
    if 'resnet' in backbone:
        if backbone == 'resnet18':
            network = resnet18
            feature_dim = 512
        elif backbone == 'resnet34':
            network = resnet34
            feature_dim = 512
        elif backbone == 'resnet50':
            network = resnet50
            feature_dim = 2048
        elif backbone == 'resnet101':
            network = resnet101
            feature_dim = 2048
        elif backbone == 'resnet152':
            network = resnet152
            feature_dim = 2048
    else:
        network = resnet50
        feature_dim = 2048

    return network, feature_dim


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return