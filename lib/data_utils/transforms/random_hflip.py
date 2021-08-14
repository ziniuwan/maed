import numpy as np
import random 
import torchvision.transforms.functional as F

from lib.data_utils.kp_utils import keypoint_2d_hflip, keypoint_3d_hflip, smpl_pose_hflip

from PIL import Image

class RandomHorizontalFlipImage(object):
    """Horizontally flip the input image, keypoints and smpl pose randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, instance):
        """
        instance (dict): must contain key 'image' and 'kp_2d'. Optional: support 'kp_3d' and 'pose' flip.

        instance['image'] is a list of PIL Images or numpy arrays.
        instance['kp_2d'] is a numpy array.
        instance['kp_3d'] is a numpy array.
        instance['pose'] is a numpy array.

        Returns:
            same as input, while image and keypoints are flipped.
        """
        if isinstance(instance['image'], Image.Image):
            image = instance['image']
        elif isinstance(instance['image'], np.ndarray):
            image = Image.fromarray(instance['image'])
        else:
            image = instance['image']
            raise TypeError(
                f'Random Horizontal Flip not yet implemented for {type(image)}')
        
        kp_2d = instance['kp_2d'].copy()
        kp_3d = instance['kp_3d'].copy() if 'kp_3d' in instance else None
        pose = instance['pose'].copy() if 'pose' in instance else None

        img_width = image.size[0]

        if random.random() < self.p:
            flipped_image = F.hflip(image)
            flipped_kp_2d = keypoint_2d_hflip(kp_2d, img_width)
            flipped_kp_3d = keypoint_3d_hflip(kp_3d) if kp_3d is not None else None
            flipped_pose = smpl_pose_hflip(pose) if pose is not None else None 
        else:
            flipped_image = image 
            flipped_kp_2d = kp_2d
            flipped_kp_3d = kp_3d
            flipped_pose = pose 

        ret = {k:v for k, v in instance.items() if k not in ['image', 'kp_2d', 'kp_3d', 'pose']}
        ret.update({'image': flipped_image, 'kp_2d':flipped_kp_2d})
        if flipped_kp_3d is not None:
            ret.update({'kp_3d': flipped_kp_3d})
        if flipped_pose is not None:
            ret.update({'pose': flipped_pose})

        return ret


class RandomHorizontalFlipVideo(object):
    """Horizontally flip the given list of PIL Images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, instance):
        """
        instance (dict): must contain key 'clip' and 'kp_2d'. Optional: support 'kp_3d' and 'pose' flip.

        instance['clip'] is a list of PIL Images or numpy arrays.
        instance['kp_2d'] is a numpy array.
        instance['kp_3d'] is a numpy array.
        instance['pose'] is a numpy array.

        Returns:
            same as input, while clip and keypoints are flipped.
        """
        if isinstance(instance['clip'][0], Image.Image):
            clip = instance['clip']
        elif isinstance(instance['clip'][0], np.ndarray):
            clip = [Image.fromarray(c) for c in instance['clip']]
        else:
            clip = instance['clip'][0]
            raise TypeError(
                f'Random Horizontal Flip not yet implemented for {type(clip)}')
        
        kp_2d = instance['kp_2d'].copy()
        kp_3d = instance['kp_3d'].copy() if 'kp_3d' in instance else None
        pose = instance['pose'].copy() if 'pose' in instance else None

        img_width = clip[0].size[0]

        if random.random() < self.p:
            flipped_clip = []
            for img in clip:
                flipped_clip.append(F.hflip(img))
            flipped_kp_2d = keypoint_2d_hflip(kp_2d, img_width)
            flipped_kp_3d = keypoint_3d_hflip(kp_3d) if kp_3d is not None else None
            flipped_pose = smpl_pose_hflip(pose) if pose is not None else None 
        else:
            flipped_clip = clip
            flipped_kp_2d = kp_2d
            flipped_kp_3d = kp_3d
            flipped_pose = pose 

        ret = {k:v for k, v in instance.items() if k not in ['clip', 'kp_2d', 'kp_3d', 'pose']}
        ret.update({'clip': flipped_clip, 'kp_2d':flipped_kp_2d})
        if flipped_kp_3d is not None:
            ret.update({'kp_3d': flipped_kp_3d})
        if flipped_pose is not None:
            ret.update({'pose': flipped_pose})

        return ret