import os
import cv2
import random

import numpy as np
import os.path as osp
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image

class _CropBase(object):
    """Crop PIL Image or numpy array according to specified bounding box.
    Also affine keypoints coordinates to make sure keypoints are on the right position of the croped clip.

    In order to make the augmentation process simple and efficient, some image-level augmention are done here
    in a coupling manner, such as random rotation and random scale jitter.

    Args:
        patch_height (float or int): cropped clip height. Default value is 224.
        patch_width (float or int): cropped clip width. Default value is 224.
        rot_jitter (float): how much to randomly rotate clip and keypoints. rotation angle
    is chosen uniformly from [-rot_jitter, rot_jitter].
        size_jitter (float): how much to randomly rescale clip and keypoints. scale factor
    is chosen uniformly from [1.3 - size_jitter, 1.3 + size_jitter].
        random_crop_p (float): how much probability to apply random crop gen_augmentation
        random_crop_size (float): max ratio of height and width to be cropped
    """
    def __init__(self, patch_height=224, patch_width=224, rot_jitter=0.,
        size_jitter=0., random_crop_p=0., random_crop_size=0.5):
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.size_jitter = size_jitter
        self.rot_jitter = rot_jitter
        self.random_crop_p = random_crop_p
        self.random_crop_size = random_crop_size

    def gen_augmentation(self):
        scale = random.uniform(1.3-self.size_jitter, 1.3+self.size_jitter)
        rot = random.uniform(-self.rot_jitter, self.rot_jitter)
        if np.random.rand() < self.random_crop_p:
            scale = np.random.uniform(1.3 - self.random_crop_size, 1.3)
            shift_w = np.random.uniform(-(1.3-scale)/2.0, (1.3-scale)/2.0)
            shift_h = np.random.uniform(-(1.3-scale)/2.0, (1.3-scale)/2.0)
            return (scale, scale), rot, (shift_w, shift_h)
        else:
            return (scale, scale), rot, (0, 0)

    def rotate_2d(self, pt_2d, rot_rad):
        x = pt_2d[0]
        y = pt_2d[1]
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy], dtype=np.float32)

    def gen_trans(self, bbox, scale, rot, shift):
        # augment size with scale
        src_w = bbox[2] * scale[0]
        src_h = bbox[3] * scale[1]
        src_center = bbox[:2] + bbox[2:] * shift

        # augment rotation
        rot_rad = np.pi * rot / 180
        src_downdir = self.rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
        src_rightdir = self.rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

        dst_w = self.patch_width
        dst_h = self.patch_height
        dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
        dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = src_center
        src[1, :] = src_center + src_downdir
        src[2, :] = src_center + src_rightdir

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_downdir
        dst[2, :] = dst_center + dst_rightdir

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def trans_image(self, image, trans):
        affined = cv2.warpAffine(image.copy(), trans, (int(self.patch_width), int(self.patch_height)),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        affined = Image.fromarray(affined)
        return affined

    def trans_keypoints(self, kp_2d, trans):
        if len(kp_2d.shape) == 1:
            # a single keypoint
            src_pt = np.array([kp_2d[0], kp_2d[1], 1.]).T
            dst_pt = np.dot(trans, src_pt)
            return np.concatenate([dst_pt[0:2], kp_2d[-1:]], axis=0)
        else:
            # list of keypoints
            new_kp = np.zeros_like(kp_2d)
            for i, kp in enumerate(kp_2d):
                new_kp[i] = self.trans_keypoints(kp, trans)
            return new_kp

    def __call__(self, instance):
        raise NotImplementedError()



class CropImage(_CropBase):
    """Crop PIL Image or numpy array and keypoints according to specified bounding box.
    """
    def __init__(self, patch_height=224, patch_width=224, rot_jitter=0.,
        size_jitter=0., random_crop_p=0., random_crop_size=0.5):
        super(CropImage, self).__init__(patch_height, patch_width, rot_jitter,
            size_jitter, random_crop_p, random_crop_size)

    def __call__(self, instance):
        if 'bbox' not in instance.keys():
            # do nothing if bbox is not specified
            return instance

        image, bbox = instance['image'], instance['bbox']
        kp_2d = instance['kp_2d'] if 'kp_2d' in instance else None

        scale, rot, shift = self.gen_augmentation()
        trans = self.gen_trans(bbox, scale, rot, shift)
        image = self.trans_image(image, trans)
        if kp_2d is not None:
            kp_2d = self.trans_keypoints(kp_2d, trans)

        ret = {k: v for k, v in instance.items() if k not in ['image', 'kp_2d']}
        ret.update({'image': image})
        if kp_2d is not None:
            ret.update({'kp_2d': kp_2d})
        return ret



class CropVideo(_CropBase):
    """Crop a sequence of PIL Image or numpy array and keypoints according to specified bounding box.
    """
    def __init__(self, patch_height=224, patch_width=224, rot_jitter=0.,
        size_jitter=0., random_crop_p=0., random_crop_size=0.5):
        super(CropVideo, self).__init__(patch_height, patch_width, rot_jitter,
            size_jitter, random_crop_p, random_crop_size)

    def __call__(self, instance):
        if 'bbox' not in instance.keys():
            # do nothing if bbox is not specified
            return instance

        clip, bboxs = instance['clip'], instance['bbox']

        kp_2d = instance['kp_2d'] if 'kp_2d' in instance else [None] * len(clip)

        scale, rot, shift = self.gen_augmentation()

        clip_croped = []
        keypoints_affine = []
        for frame, bbox, keypoint in zip(clip, bboxs, kp_2d):
            trans = self.gen_trans(bbox, scale, rot, shift)
            clip_croped.append(self.trans_image(frame, trans))
            if keypoint is not None:
                keypoints_affine.append(self.trans_keypoints(keypoint, trans))

        if len(keypoints_affine) > 0:
            keypoints_affine = np.stack(keypoints_affine, axis=0)

        ret = {k: v for k, v in instance.items() if k not in ['clip', 'kp_2d']}
        ret.update({'clip': clip_croped})
        if len(keypoints_affine) > 0:
            ret.update({'kp_2d': keypoints_affine})
        return ret
