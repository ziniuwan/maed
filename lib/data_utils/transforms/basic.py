import torch
import torchvision.transforms.functional as F
import numpy as np

from PIL import Image

class _NormalizeBase(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], patch_size=224, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.patch_size = patch_size

    def normalize_2d_kp(self, kp_2d):
        # Normalize keypoints between -1, 1
        ratio = 1.0 / self.patch_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0

        return kp_2d

    def __call__(self, instance):
        raise NotImplementedError()

class NormalizeVideo(_NormalizeBase):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], patch_size=224, inplace=False):
        super(NormalizeVideo, self).__init__(mean, std, patch_size, inplace)

    def __call__(self, instance):
        clip = instance['clip']
        new_clip = []
        for c in clip:
            new_clip.append(F.normalize(c, self.mean, self.std, self.inplace))
        new_clip = torch.stack(new_clip, dim=0)

        ret = {k: v for k, v in instance.items() if k not in ['clip', 'kp_2d', 'kp_2d_full']}
        ret.update({'clip': new_clip})

        if 'kp_2d' in instance:
            kp = instance['kp_2d']
            new_kp = kp
            new_kp[:,:, :2] = self.normalize_2d_kp(kp[:,:, :2])
            ret.update({'kp_2d':new_kp})
        
        if 'kp_2d_full' in instance:
            kp = instance['kp_2d_full']
            new_kp = kp
            new_kp[:,:, :2] = self.normalize_2d_kp(kp[:,:, :2])
            ret.update({'kp_2d_full':new_kp})

        return ret

class NormalizeImage(_NormalizeBase):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], patch_size=224, inplace=False):
        super(NormalizeImage, self).__init__(mean, std, patch_size, inplace)

    def __call__(self, instance):
        image = instance['image']
        new_image = F.normalize(image, self.mean, self.std, self.inplace)

        if 'kp_2d' in instance:
            kp = instance['kp_2d']
            new_kp = kp
            new_kp[:, :2] = self.normalize_2d_kp(kp[:, :2])
        
        ret = {k: v for k, v in instance.items() if k not in ['image', 'kp_2d']}

        if 'kp_2d' in instance:
            ret.update({'kp_2d':new_kp})

        ret.update({'image': new_image})

        return ret

class StackFrames(object):
    """Stack a list of PIL Images or numpy arrays along a new dimension.

    Args:
        roll (float): whether to convert BGR to RGB. Default value is False
    """
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, instance):
        clip = instance['clip']
        if self.roll:
            stacked_clip = np.stack([np.array(x)[:, :, ::-1] for x in clip], axis=0)
        else:
            stacked_clip = np.stack([np.array(x) for x in clip], axis=0)

        ret = {k:v for k, v in instance.items() if k!='clip'}
        ret.update({'clip': stacked_clip})
        return ret

class ToTensorVideo(object):
    """ Converts a sequence of PIL.Image (RGB) or numpy.ndarray (T x H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (T x C x H x W) in the range [0.0, 1.0] """
    def __call__(self, instance):
        clip = instance['clip']
        new_clip = []
        for img in clip:
            img = F.to_tensor(img)
            new_clip.append(img)
        clip = torch.stack(new_clip, dim=0)

        ret = {k: torch.from_numpy(v) for k, v in instance.items() if k!='clip'}
        ret.update({'clip': clip})
        return ret

class ToTensorImage(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __call__(self, instance):
        image = instance['image']
        image = F.to_tensor(image)
        ret = {k: torch.from_numpy(v) for k, v in instance.items() if k!='image'}
        ret.update({'image': image})
        return ret