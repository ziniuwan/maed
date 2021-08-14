# -*- coding: utf-8 -*-
"""
This script is borrowed from https://github.com/mkocabas/VIBE.
Adhere to their license to use this script.

We hacked it a little bit to make it happy in our framework.
"""

import sys
sys.path.append('.')

import glob
import torch
import joblib
import argparse
from tqdm import tqdm
import os.path as osp
from skimage import io
from scipy.io import loadmat

from lib.data_utils.kp_utils import *
from lib.core.config import DB_DIR, PENNACTION_DIR
from lib.data_utils.img_utils import get_bbox_from_kp2d


def calc_kpt_bound(kp_2d):
    MAX_COORD = 10000
    x = kp_2d[:, 0]
    y = kp_2d[:, 1]
    z = kp_2d[:, 2]
    u = MAX_COORD
    d = -1
    l = MAX_COORD
    r = -1
    for idx, vis in enumerate(z):
        if vis == 0:  # skip invisible joint
            continue
        u = min(u, y[idx])
        d = max(d, y[idx])
        l = min(l, x[idx])
        r = max(r, x[idx])
    return u, d, l, r


def load_mat(path):
    mat = loadmat(path)
    del mat['pose'], mat['__header__'], mat['__globals__'], mat['__version__'], mat['train'], mat['action']
    mat['nframes'] = mat['nframes'][0][0]

    return mat


def read_data(folder):
    dataset = {
        'img_name' : [],
        'joints2D': [],
        'bbox': [],
        'vid_name': [],
    }

    file_names = sorted(glob.glob(folder + '/labels/'+'*.mat'))

    for fname in tqdm(file_names):
        vid_dict=load_mat(fname)
        imgs = sorted(glob.glob(folder + '/frames/'+ fname.strip().split('/')[-1].split('.')[0]+'/*.jpg'))
        kp_2d = np.zeros((vid_dict['nframes'], 13, 3))
        perm_idxs = get_perm_idxs('pennaction', 'common')

        kp_2d[:, :, 0] = vid_dict['x']
        kp_2d[:, :, 1] = vid_dict['y']
        kp_2d[:, :, 2] = vid_dict['visibility']
        kp_2d = kp_2d[:, perm_idxs, :]

        # fix inconsistency
        n_kp_2d = np.zeros((kp_2d.shape[0], 14, 3))
        n_kp_2d[:, :12, :] = kp_2d[:, :-1, :]
        n_kp_2d[:, 13, :] = kp_2d[:, 12, :]
        kp_2d = n_kp_2d

        bbox = np.zeros((vid_dict['nframes'], 4))

        for fr_id, fr in enumerate(kp_2d):
            u, d, l, r = calc_kpt_bound(fr)
            center = np.array([(l + r) * 0.5, (u + d) * 0.5], dtype=np.float32)
            c_x, c_y = center[0], center[1]
            w, h = r - l, d - u
            w = h = np.where(w / h > 1, w, h)

            bbox[fr_id,:] = np.array([c_x, c_y, w, h])

        dataset['vid_name'].append(np.array([f'{fname}']* vid_dict['nframes']))
        dataset['img_name'].append(np.array(imgs))
        dataset['joints2D'].append(kp_2d)
        dataset['bbox'].append(bbox)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
        dataset[k] = np.concatenate(dataset[k])
    
    dataset['joints2D'] = convert_kps(dataset['joints2D'], src='pennaction', dst='spin')

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir', type=str, help='dataset directory', default=PENNACTION_DIR)
    parser.add_argument('--out_dir', type=str, help='output directory', default=DB_DIR)
    args = parser.parse_args()

    dataset = read_data(args.inp_dir)
    joblib.dump(dataset, osp.join(args.out_dir, 'pennaction_train_db.pt'))