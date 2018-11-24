import cv2 as cv

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def non_maximum_suppression(positions, index, query_num, distance_threshold):
    """
    avoid redundant results that are nearby in positions
    :param positions:  N * 2
    :param index: rank index,
    :param query_num:
    :param distance_threshold:
    :return: first query_num indices
    """
    N = len(index)
    assert query_num < N

    result = []

    for i in range(N):
        # compare candidate location with selected results
        p = positions[index[i], :]
        is_valid = True
        for j in range(len(result)):
            q = positions[result[j], :]
            dis = np.linalg.norm(p - q)
            if dis < distance_threshold:
                is_valid = False
                break
        if is_valid:
            result.append(index[i])

        if len(result) == query_num:
            break
    return result

def crop_image_patch(im, position, patch_size):
    """
    crop image patch centered in position
    :param im:
    :param position: patch center position,
                     not close to image border
    :param patch_size:
    :return:
    """
    h, w = im.shape[0], im.shape[1]
    delta = patch_size/2
    x = int(position[0] - delta)
    y = int(position[1] - delta)
    x2, y2 = x + patch_size, y + patch_size
    assert x >= 0 and x < w and y >= 0 and y < h
    assert x2 >= 0 and x2 < w
    assert y2 >= 0 and y2 < h

    im_patch = im[y:y2, x:x2]
    return im_patch

def vis_images(images, cols):
    """
    :param images: list of images
    :param cols:
    :return:
    """
    assert cols > 0
    N = len(images)
    rows = N//cols
    if N%cols != 0:
        rows += 1
    fig = plt.figure(figsize=(rows,cols))
    for r in range(rows):
        for c in range(cols):
            index = r * cols + c
            if index < N:
                pass
                fig.add_subplot(rows, cols, index + 1)
                plt.imshow(images[index][:,:,[2,1,0]]) # RGB to BGR as in opencv
                plt.axis('equal')
                plt.axis('off')
    plt.show()

def vis_cadindate_location(im, points):
    """

    :param im:  opencv color image
    :param points: N * 2
    :return:
    """
    vis = im.copy()
    for r in range(points.shape[0]):
        x, y = points[r,:]
        cv.circle(vis, (int(x), int(y)), 5, (255, 0, 0), 2)
    #cv.imshow('candidate locations', vis)
    #cv.waitKey()
    fig = plt.figure()
    plt.imshow(vis[:,:,[2,1,0]])
    plt.axis('equal')
    plt.axis('off')
    plt.show(block=False)
