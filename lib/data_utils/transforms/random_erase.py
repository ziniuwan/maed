import random
import numpy as np

from PIL import Image

class _RandomEraseBase(object):
    """Randomly erase the lower part of the clip
    Args:
    prob (float): The probability to apply random erase
    max_erase_part (float): The maximum ratio of the erase part
    random_filling (bool): if True, fill the erased part with random pixel, otherwise with zero pixel.
    erase_kp (bool): if True, mask out the keypoints in the erased part.
    margin: (float): if <erase_kp> is set to True, the keypoints in the margin between erased part and unerased part will not be masked out. 
    """

    def __init__(self, prob=0, max_erase_part=0.5, random_filling=True, erase_kp=True, margin=0.1):
        self.prob = prob
        self.max_erase_part = max_erase_part
        self.random_filling = random_filling
        self.erase_kp = erase_kp
        self.margin = margin
    
    def _erase_top(self, img, kp_2d, kp_3d, erased_ratio):
        h,w,_ = img.shape
        erased_h = int(h * erased_ratio)
        if erased_h > 0:
            if self.random_filling:
                img[:erased_h] = np.random.randint(256, size=(erased_h, w, 3), dtype=np.uint8)
            else:
                img[:erased_h] = 0
            if self.erase_kp:
                for i, kp in enumerate(kp_2d):
                    if erased_h - kp[1] > h * self.margin:
                        kp_2d[2] = 0. 
                        if kp_3d is not None:
                            kp_3d[t, i, -1] = 0
        return img, kp_2d, kp_3d
    
    def _erase_bottom(self, img, kp_2d, kp_3d, erased_ratio):
        h,w,_ = img.shape
        erased_h = int(h * erased_ratio)
        if erased_h > 0:
            if self.random_filling:
                img[-erased_h:] = np.random.randint(256, size=(erased_h, w, 3), dtype=np.uint8)
            else:
                img[-erased_h:] = 0
            if self.erase_kp:
                for i, kp in enumerate(kp_2d):
                    if erased_h - (h - kp[1]) > h * self.margin:
                        kp_2d[2] = 0. 
                        if kp_3d is not None:
                            kp_3d[t, i, -1] = 0
        return img, kp_2d, kp_3d
    
    def _erase_left(self, img, kp_2d, kp_3d, erased_ratio):
        h,w,_ = img.shape
        erased_w = int(w * erased_ratio)
        if erased_w > 0:
            if self.random_filling:
                img[:erased_w] = np.random.randint(256, size=(h, erased_w, 3), dtype=np.uint8)
            else:
                img[:erased_w] = 0
            if self.erase_kp:
                for i, kp in enumerate(kp_2d):
                    if erased_w - kp[0] > w * self.margin:
                        kp_2d[2] = 0. 
                        if kp_3d is not None:
                            kp_3d[t, i, -1] = 0
        return img, kp_2d, kp_3d
    
    def _erase_right(self, img, kp_2d, kp_3d, erased_ratio):
        h,w,_ = img.shape
        erased_w = int(w * erased_ratio)
        if erased_w > 0:
            if self.random_filling:
                img[-erased_w:] = np.random.randint(256, size=(h, erased_w, 3), dtype=np.uint8)
            else:
                img[-erased_w:] = 0
            if self.erase_kp:
                for i, kp in enumerate(kp_2d):
                    if erased_w - (w - kp[0]) > w * self.margin:
                        kp_2d[2] = 0. 
                        if kp_3d is not None:
                            kp_3d[t, i, -1] = 0
        return img, kp_2d, kp_3d

    def __call__(self, instance):
        raise NotImplementedError()


class RandomEraseVideo(_RandomEraseBase):
    """Randomly erase the lower part of the clip
    Args:
    prob (float): The probability to apply random erase
    max_erase_part (float): The maximum ratio of the erase part
    random_filling (bool): if True, fill the erased part with random pixel, otherwise with zero pixel.
    erase_kp (bool): if True, mask out the keypoints in the erased part.
    margin: (float): if <erase_kp> is set to True, the keypoints in the margin between erased part and unerased part will not be masked out. 
    """

    def __init__(self, prob=0, max_erase_part=0.5, random_filling=True, erase_kp=True, margin=0.1):
        super(RandomEraseVideo, self).__init__(prob, max_erase_part, random_filling, erase_kp, margin)

    def __call__(self, instance):
        """
        Args:
        instance (dict): must contain key 'image'. 
        instance['image'] PIL Image or numpy arrays.
        """
        if isinstance(instance['clip'][0], Image.Image):
            clip = instance['clip']
        elif isinstance(instance['clip'][0], np.ndarray):
            clip = [Image.fromarray(c) for c in instance['clip']]
        else:
            clip = instance['clip'][0]
            raise TypeError(
                f'Random Erase not yet implemented for {type(clip)}')

        kp_2d = instance['kp_2d'].copy()
        kp_3d = instance['kp_3d'].copy() if 'kp_3d' in instance else None

        # Apply to all images
        erased_clip = []
        erased_kp_2ds = []

        erased_part = random.choice([self._erase_left, self._erase_right, self._erase_top, self._erase_bottom])
        for t, (kp_2d_frame, img) in enumerate(zip(kp_2d, clip)):
            erased_img = img.copy()
            erased_kp_2d = kp_2d_frame.copy()
            if np.random.rand() < self.prob:
                erased_ratio = np.random.rand() * self.max_erase_part
                erased_img = np.array(erased_img)
                erased_img, erased_kp_2d, _ = erased_part(erased_img, erased_kp_2d, None, erased_ratio)

                erased_img = Image.fromarray(erased_img)
            
            erased_kp_2ds.append(erased_kp_2d)
            erased_clip.append(erased_img)

        erased_kp_2ds = np.stack(erased_kp_2ds, axis=0)


        ret = {k:v for k, v in instance.items() if k not in ['clip', 'kp_2d', 'kp_3d']}
        ret.update({'clip': erased_clip, 'kp_2d': erased_kp_2ds})
        if kp_3d is not None:
            ret.update({'kp_3d': kp_3d})


        return ret


class RandomEraseImage(_RandomEraseBase):
    """Randomly erase the lower part of the clip
    Args:
    prob (float): The probability to apply random erase
    max_erase_part (float): The maximum ratio of the erase part
    random_filling (bool): if True, fill the erased part with random pixel, otherwise with zero pixel.
    erase_kp (bool): if True, mask out the keypoints in the erased part.
    margin: (float): if <erase_kp> is set to True, the keypoints in the margin between erased part and unerased part will not be masked out. 
    """

    def __init__(self, prob=0, max_erase_part=0.5, random_filling=True, erase_kp=True, margin=0.1):
        super(RandomEraseImage, self).__init__(prob, max_erase_part, random_filling, erase_kp, margin)

    def __call__(self, instance):
        """
        Args:
        instance (dict): must contain key 'image'. 
        instance['image'] PIL Image or numpy arrays.
        """
        if isinstance(instance['image'], Image.Image):
            image = instance['image']
        elif isinstance(instance['image'], np.ndarray):
            image = Image.fromarray(instance['image'])
        else:
            image = instance['image']
            raise TypeError(
                f'Random Erase not yet implemented for {type(image)}')

        kp_2d = instance['kp_2d'].copy()
        kp_3d = instance['kp_3d'].copy() if 'kp_3d' in instance else None

        erased_part = random.choice([self._erase_left, self._erase_right, self._erase_top, self._erase_bottom])
        erased_img = image.copy()
        erased_kp_2d = kp_2d.copy()
        if np.random.rand() < self.prob:
            erased_ratio = np.random.rand() * self.max_erase_part
            erased_img = np.array(erased_img)
            erased_img, erased_kp_2d, _ = erased_part(erased_img, kp_2d, None, erased_ratio)
            erased_img = Image.fromarray(erased_img)


        ret = {k:v for k, v in instance.items() if k not in ['image', 'kp_2d', 'kp_3d']}
        ret.update({'image': erased_img, 'kp_2d': erased_kp_2d})
        if kp_3d is not None:
            ret.update({'kp_3d': kp_3d})

        return ret