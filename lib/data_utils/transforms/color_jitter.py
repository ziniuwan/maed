import random
import torchvision.transforms.functional as F
import numpy as np

from PIL import Image


class _ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

class ColorJitterVideo(_ColorJitter):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitterVideo, self).__init__(brightness, contrast, saturation, hue)

    def __call__(self, instance):
        """
        Args:
            instance (dict): must contain key 'clip', and instance['clip'] is a list of PIL Image or numpy array
        """
        if isinstance(instance['clip'][0], Image.Image):
            clip = instance['clip']
        elif isinstance(instance['clip'][0], np.ndarray):
            clip = [Image.fromarray(c) for c in instance['clip']]
        else:
            clip = instance['clip'][0]
            raise TypeError(
                f'Color jitter not yet implemented for {type(clip)}')

        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: F.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: F.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: F.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: F.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)
        # Apply to all images
        jittered_clip = []
        for img in clip:
            jittered_img = img.copy()
            for func in img_transforms:
                jittered_img = func(jittered_img)
            jittered_clip.append(jittered_img)


        ret = {k:v for k, v in instance.items() if k!='clip'}
        ret.update({'clip': jittered_clip})

        return ret

class ColorJitterImage(_ColorJitter):
    """Randomly change the brightness, contrast and saturation and hue of the images
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitterImage, self).__init__(brightness, contrast, saturation, hue)

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

        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: F.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: F.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: F.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: F.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        # Apply to images
        jittered_img = image.copy()
        for func in img_transforms:
            jittered_img = func(jittered_img)

        ret = {k:v for k, v in instance.items() if k!='image'}
        ret.update({'image': jittered_img})

        return ret