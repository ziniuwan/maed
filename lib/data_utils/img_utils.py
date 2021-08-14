import os
import cv2
import torch
import mc 
import io 
import numpy as np
import os.path as osp

from skimage.util.shape import view_as_windows
from PIL import Image

def get_bbox_from_kp2d(kp_2d):
    # get bbox
    if len(kp_2d.shape) > 2:
        ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
        lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    else:
        ul = np.array([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = np.array([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    # ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)
    w = h = h * 1.1

    bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    return bbox

def split_into_chunks(vid_names, seqlen, stride, pad=True):
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if pad:
            padlen = (seqlen - indexes.shape[0] % seqlen) % seqlen
            indexes = np.pad(indexes, ((0, padlen)), 'reflect')
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        chunks = chunks.tolist()
        #start_finish = chunks[:, (0, -1)].tolist()
        #video_start_end_indices += start_finish
        video_start_end_indices += chunks

    return video_start_end_indices

def pad_image(img, h, w):
    img = img.copy()
    img_h, img_w, _ = img.shape
    pad_top = (h - img_h) // 2
    pad_bottom = h - img_h - pad_top 
    pad_left = (w - img_w) // 2
    pad_right = w - img_w - pad_left

    img = np.pad(img, ((pad_top, pad_bottom),(pad_left, pad_right),(0, 0)))

    return img

def read_img(path, convert='RGB', check_exist=False):
    if check_exist and not osp.exists(path):
        return None
    try:
        img = Image.open(path)
        if convert:
            img = img.convert(convert)
    except:
        raise IOError('File error: ', path)
    return np.array(img)
