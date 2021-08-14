# -*- coding: utf-8 -*-
import os
import torch
import logging
import numpy as np
import os.path as osp
import joblib
import random

from torch.utils.data import Dataset

from lib.core.config import DB_DIR
from lib.models.smpl import OP_TO_J14, J49_TO_J14
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import split_into_chunks, read_img

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, dataset_name, set, transforms,
        seqlen=0, overlap=0., sample_pool=64, 
        random_sample=True, random_start=False,
        pad=True, verbose=True, debug=False):

        self.dataset_name = dataset_name
        self.set = set
        self.transforms = transforms
        
        assert seqlen > 0 or sample_pool > 0
        self.seqlen = seqlen if seqlen > 0 else sample_pool
        self.sample_pool = sample_pool if sample_pool > 0 else seqlen
        self.sample_freq = self.sample_pool // self.seqlen
        #assert self.sample_pool % self.seqlen == 0

        self.overlap = overlap
        self.stride = max(int(self.sample_pool * (1-overlap)), 1) if overlap < 1 else overlap

        self.random_sample = random_sample
        self.random_start = random_start
        assert not (self.random_sample and self.random_start)
        # Either random sample or random start, cannot be both

        self.debug = debug
        self.verbose = verbose
        
        self.db = self._load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.sample_pool, self.stride, pad)

        if self.verbose:
            print(f'{self.dataset_name} - Dataset overlap ratio: {self.overlap}')
            print(f'{self.dataset_name} - Number of dataset objects {self.__len__()}')


    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        is_train = self.set == 'train'
        target = {}
        
        # determine sample index
        sample_idx, full_sample_idx = self.gen_sample_index(index)

        # load and process 2D&3D keypoints
        kp_2d, kp_3d = self.get_keypoints(sample_idx)

        # load SMPL parameters: theta, beta along with cam params. 
        cam, pose, shape, w_smpl = self.get_smpl_params(sample_idx)
        target['w_smpl'] = w_smpl
        
        # bounding box
        if self.dataset_name != 'insta':
            bbox = self.db['bbox'][sample_idx]
            if not is_train:
                target['bbox'] = self.db['bbox'][sample_idx]
        
        # images
        image_paths = self.db['img_name'][sample_idx]
        images = [read_img(path) for path in image_paths]

        if not is_train:
            target['paths'] = self.db['img_name'][sample_idx].tolist()

        # preprocess and augmentation
        raw_inp = {
            'clip': images, 
            'kp_2d': kp_2d,  
            'kp_3d':kp_3d, 
            'pose':pose,
            'shape':shape,
            'cam':cam,
        }
        if self.dataset_name != 'insta':
            raw_inp['bbox'] = bbox 
        transformed = self.transforms(raw_inp)

        target['images'] = transformed['clip'].float()
        target['kp_2d'] = transformed['kp_2d'].float()
        target['kp_3d'] = transformed['kp_3d'].float()

        theta = torch.cat([transformed['cam'].float(), transformed['pose'].float(), transformed['shape'].float()], dim=1) #(T, 85)
        target['theta'] = theta.float() # camera, pose and shape

        # optional info for evaluation
        if self.dataset_name == 'mpii3d' and not is_train:
            target['valid'] = self.db['valid_i'][sample_idx]
            vn = self.db['vid_name'][sample_idx]
            fi = self.db['frame_id'][sample_idx]
            target['instance_id'] = [f'{v}/{f}'for v,f in zip(vn,fi)]
        if self.dataset_name in ['3dpw', 'h36m'] and not is_train:
            vn = self.db['vid_name'][sample_idx]
            fi = self.db['frame_id'][sample_idx]
            target['instance_id'] = [f'{v}/{f}'for v,f in zip(vn,fi)]
        if not is_train:
            valid = np.array(full_sample_idx)
            valid = valid - np.roll(valid, 1)
            valid = valid > 0
            valid[0] = True
            target['valid'] = torch.from_numpy(valid)

        # record data source for further use
        target['index'] = torch.tensor([index])

        return target

    def _load_db(self):
        db_file = osp.join(DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        
        if self.verbose:
            print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db
    
    def gen_sample_index(self, index):
        full_sample_idx = self.vid_indices[index]

        if self.random_sample:
            sample_idx = []
            for i in range(self.seqlen):
                sample_idx.append(full_sample_idx[self.sample_freq*i + random.randint(0, self.sample_freq-1)])
        elif self.random_start:
            start = random.randint(0, self.sample_freq-1)
            sample_idx = full_sample_idx[start::self.sample_freq][:self.seqlen]
        else:
            sample_idx = full_sample_idx[::self.sample_freq][:self.seqlen]

        return sample_idx, full_sample_idx
    
    def get_keypoints(self, sample_idx):
        if 'joints2D' in self.db:
            kp_2d = self.db['joints2D'][sample_idx]
        else:
            kp_2d = np.zeros([self.seqlen, 49, 3])

        if 'joints3D' in self.db:
            kp_3d = self.db['joints3D'][sample_idx]
        else:
            kp_3d = np.zeros([self.seqlen, 49, 4])
        
        return kp_2d, kp_3d

    def get_smpl_params(self, sample_idx):
        # w_smpl indicates whether the instance's SMPL parameters are valid
        if 'pose' in self.db:
            assert 'shape' in self.db
            pose  = self.db['pose'][sample_idx]
            shape = self.db['shape'][sample_idx]
            w_smpl = torch.ones(self.seqlen).float()
        else:
            pose = np.zeros((self.seqlen, 72))
            shape = np.zeros((self.seqlen, 10))
            w_smpl = torch.zeros(self.seqlen).float()
        
        cam = np.concatenate([np.ones((self.seqlen, 1)), np.zeros((self.seqlen, 2))], axis=1) #(T, 3)
        return cam, pose, shape, w_smpl
