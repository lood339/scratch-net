import torchvision.transforms as transforms
import torch.nn as nn
import torch

import cv2 as cv
from PIL import Image
import numpy as np

def extract_deep_feature(model, im):
    """
    :param model:
    :param im: image read from opencv
    :return: N * D, each row is L2 normalized
    """
    assert isinstance(model, nn.Module)
    assert len(im.shape) == 3 and im.shape[2] == 3

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = Image.fromarray(im)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    input = transform(im)
    input = input.view(-1, 3, 224, 224)

    feature = model.feature_avg_pool(input)
    feature = feature.data
    feature = feature.numpy()
    feature = np.squeeze(feature, axis=(2, 3))

    feature_norm = np.linalg.norm(feature, axis=1)
    feature = feature / feature_norm[:, np.newaxis]
    return feature

def extract_deep_feature_batch(model, images):
    """
    :param model:
    :param images:
    :return:
    """
    assert isinstance(images, list)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    N = len(images)
    assert N > 0
    # N x c x h x w tensor
    H, W, C = 224, 224, 3
    input = torch.zeros((N, C, H, W))
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    for i, im in enumerate(images):
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        im = transform(im)
        input[i] = im

    input = input.view(-1, 3, 224, 224)

    feature = model.feature_avg_pool(input)
    feature = feature.data
    feature = feature.numpy()
    feature = np.squeeze(feature, axis=(2, 3))

    feature_norm = np.linalg.norm(feature, axis=1)
    feature = feature / feature_norm[:, np.newaxis]
    return feature