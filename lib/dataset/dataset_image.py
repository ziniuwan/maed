# -*- coding: utf-8 -*-
import os
import mc
import cv2
import torch
import numpy as np
import joblib
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset

from lib.utils.geometry import rotation_matrix_to_angle_axis
from lib.data_utils.img_utils import read_img
from lib.core.config import DB_DIR


class ImageDataset(Dataset):
    def __init__(self, dataset_name, set, 
        transforms=None, 
        verbose=True, debug=False):

        self.dataset_name = dataset_name
        self.set = set
        self.transforms = transforms

        self.debug = debug
        self.verbose = verbose
        
        self.db = self._load_db()

        if self.verbose:
            print(f'{self.dataset_name} - Number of dataset objects {self.__len__()}')

    def _load_db(self):
        db_file = osp.join(DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        
        if self.verbose:
            print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def __len__(self):
        return len(self.db['img_name'])

    def __getitem__(self, index):
        kp_2d = self.db['joints2D'][index]
        kp_3d = self.db['joints3D'][index] if 'joints3D' in self.db else np.zeros([49, 4])
        image = read_img(self.db['img_name'][index])
        shape = self.db['shape'][index] if 'shape' in self.db else np.zeros([10])
        cam = self.db['cam'][index] if 'cam' in self.db else np.array([1., 0., 0.])
        bbox = self.db['bbox'][index]

        pose = self.db['pose'][index].astype(np.float32) if 'pose' in self.db else np.zeros([72])
        if len(pose.shape) > 1:
            pose = rotation_matrix_to_angle_axis(torch.from_numpy(pose)).numpy().flatten()

        target = {
            'image': image,
            'kp_2d': kp_2d,
            'kp_3d': kp_3d,
            'pose':pose,
            'shape':shape,
            'cam':cam,
            'bbox': bbox
        }
        if self.transforms:
            target = self.transforms(target)
        
        target['theta'] = torch.cat([target['cam'].float(), target['pose'].float(), target['shape'].float()], dim=0) # camera, pose and shape
        target['w_smpl'] = torch.tensor(1).float() if 'pose' in self.db else torch.tensor(0).float()
        
        new_target = {}
        for k, v in target.items():
            if k in ['pose', 'cam', 'shape']:
                continue
            new_target[k] = v.float()

        return new_target
