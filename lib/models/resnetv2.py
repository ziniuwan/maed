""" 
This script is borrowed from https://github.com/rwightman/pytorch-image-models.
Adhere to their license to use this script

We hacked it a little bit to make it happy in our framework.
"""

from collections import OrderedDict  # pylint: disable=g-importing-member

import logging
import torch
import math
import os
import torch.nn as nn
from functools import partial
from itertools import repeat

from torch.nn import functional as F
from .ops import DropPath

import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url, download_url_to_file, urlparse, HASH_REGEX
try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

_logger = logging.getLogger(__name__)

model_urls = {
    'resnetv2_50x1_bitm_in21k': 'https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz',
    'resnetv2_50x1_vit': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth'
}

class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if act_layer is not None and apply_act:
            self.act = act_layer(inplace=inplace)
        else:
            self.act = None

    def forward(self, x):
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        if self.act is not None:
            x = self.act(x)
        return x

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def pad_same(x, k, s, d= (1, 1), value=0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D max pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False, count_include_pad=True):
        kernel_size = tuple(repeat(kernel_size, 2))
        stride = tuple(repeat(stride, 2))
        dilation = tuple(repeat(dilation, 2))
        super(MaxPool2dSame, self).__init__(kernel_size, stride, (0, 0), dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float('inf'))
        return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)

class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.
    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False, eps=1e-5):
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=0, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return weight

    def forward(self, x):
        x = pad_same(x, self.get_weight().shape[-2:], self.stride, self.dilation)
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, (0, 0), self.dilation, self.groups)


def make_div(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
            self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1,
            act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.):
        super().__init__()
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2dSame
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, preact=True,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x_preact = self.norm1(x)

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        x = self.drop_path(x)
        return x + shortcut


class Bottleneck(nn.Module):
    """Non Pre-activation bottleneck block, equiv to V1.5/V1b Bottleneck. Used for ViT.
    """
    def __init__(
            self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1,
            act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.):
        super().__init__()
        first_dilation = first_dilation or dilation
        act_layer = act_layer or nn.ReLU
        conv_layer = conv_layer or StdConv2dSame
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, preact=False,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm1 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm2 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.norm3 = norm_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.act3 = act_layer(inplace=True)

    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop_path(x)
        x = self.act3(x + shortcut)
        return x


class DownsampleConv(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, preact=True,
            conv_layer=None, norm_layer=None):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(x))

class ResNetStage(nn.Module):
    """ResNet Stage."""
    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio=0.25, groups=1,
                 block_dpr=None, block_fn=PreActBottleneck,
                 act_layer=None, conv_layer=None, norm_layer=None, **block_kwargs):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.
            stride = stride if block_idx == 0 else 1
            self.blocks.add_module(str(block_idx), block_fn(
                prev_chs, out_chs, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups,
                first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate,
                **layer_kwargs, **block_kwargs))
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self, x):
        x = self.blocks(x)
        return x


def create_stem(in_chs, out_chs, stem_type='', preact=True, conv_layer=None, norm_layer=None):
    stem = OrderedDict()
    assert stem_type in ('', 'fixed', 'same', 'deep', 'deep_fixed', 'deep_same')

    # NOTE conv padding mode can be changed by overriding the conv_layer def
    if 'deep' in stem_type:
        # A 3 deep 3x3  conv stack as in ResNet V1D models
        mid_chs = out_chs // 2
        stem['conv1'] = conv_layer(in_chs, mid_chs, kernel_size=3, stride=2)
        stem['conv2'] = conv_layer(mid_chs, mid_chs, kernel_size=3, stride=1)
        stem['conv3'] = conv_layer(mid_chs, out_chs, kernel_size=3, stride=1)
    else:
        # The usual 7x7 stem conv
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)

    if not preact:
        stem['norm'] = norm_layer(out_chs)

    if 'fixed' in stem_type:
        # 'fixed' SAME padding approximation that is used in BiT models
        stem['pad'] = nn.ConstantPad2d(1, 0.)
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    elif 'same' in stem_type:
        # full, input size based 'SAME' padding, used in ViT Hybrid model
        stem['pool'] = MaxPool2dSame(kernel_size=3, stride=2)
    else:
        # the usual PyTorch symmetric padding
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    return nn.Sequential(stem)


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """

    def __init__(self, layers, channels=(256, 512, 1024, 2048),
                 num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
                 width_factor=1, stem_chs=64, stem_type='', preact=True,
                 act_layer=nn.ReLU, conv_layer=StdConv2dSame, norm_layer=partial(GroupNormAct, num_groups=32),
                 drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor

        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_stem(in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        # NOTE no, reduction 2 feature if preact
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module='' if preact else 'stem.norm'))

        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(
                prev_chs, out_chs, stride=stride, dilation=dilation, depth=d,
                act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            prev_chs = out_chs
            curr_stride *= stride
            feat_name = f'stages.{stage_idx}'
            if preact:
                feat_name = f'stages.{stage_idx + 1}.blocks.0.norm1' if (stage_idx + 1) != len(channels) else 'norm'
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=feat_name)]
            self.stages.add_module(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else nn.Identity()
        if global_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.global_pool = nn.Identity()
        self.head = nn.Identity()

        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or ('.fc' in n and isinstance(m, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward(self, x, seqlen=8):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.squeeze(2).squeeze(2)
        x = self.head(x)
        return x

    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        import numpy as np
        weights = np.load(checkpoint_path)
        with torch.no_grad():
            stem_conv_w = tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel'])
            if self.stem.conv.weight.shape[1] == 1:
                self.stem.conv.weight.copy_(stem_conv_w.sum(dim=1, keepdim=True))
                # FIXME handle > 3 in_chans?
            else:
                self.stem.conv.weight.copy_(stem_conv_w)
            self.norm.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.norm.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            #self.head.fc.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))
            #self.head.fc.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))
            for i, (sname, stage) in enumerate(self.stages.named_children()):
                for j, (bname, block) in enumerate(stage.blocks.named_children()):
                    convname = 'standardized_conv2d'
                    block_prefix = f'{prefix}block{i + 1}/unit{j + 1:02d}/'
                    block.conv1.weight.copy_(tf2th(weights[f'{block_prefix}a/{convname}/kernel']))
                    block.conv2.weight.copy_(tf2th(weights[f'{block_prefix}b/{convname}/kernel']))
                    block.conv3.weight.copy_(tf2th(weights[f'{block_prefix}c/{convname}/kernel']))
                    block.norm1.weight.copy_(tf2th(weights[f'{block_prefix}a/group_norm/gamma']))
                    block.norm2.weight.copy_(tf2th(weights[f'{block_prefix}b/group_norm/gamma']))
                    block.norm3.weight.copy_(tf2th(weights[f'{block_prefix}c/group_norm/gamma']))
                    block.norm1.bias.copy_(tf2th(weights[f'{block_prefix}a/group_norm/beta']))
                    block.norm2.bias.copy_(tf2th(weights[f'{block_prefix}b/group_norm/beta']))
                    block.norm3.bias.copy_(tf2th(weights[f'{block_prefix}c/group_norm/beta']))
                    if block.downsample is not None:
                        w = weights[f'{block_prefix}a/proj/{convname}/kernel']
                        block.downsample.conv.weight.copy_(tf2th(w))



def resnetv2_50x1_bitm_in21k(pretrained=False, **kwargs):
    model = ResNetV2(layers=[3, 4, 6, 3], width_factor=1, stem_type='fixed', **kwargs)
    if pretrained:
        load_custom_pretrained(model, model_urls['resnetv2_50x1_bitm_in21k'])
    return model

def resnetv2_50x1_vit(pretrained=False, strict=False, progress=False, **kwargs):
    """ ResNetv2-50 from ViT-B/16 hybrid model from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    model = ResNetV2(
        layers=(3, 4, 9), num_classes=0, global_pool='avg', in_chans=kwargs.get('in_chans', 3),
        preact=False, stem_type='same')
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnetv2_50x1_vit'], progress=progress, map_location='cpu')
        state_dict = {'.'.join(k.split('.')[2:]):v for k, v in state_dict.items() if k.startswith('patch_embed.backbone')}
        miss, unexp = model.load_state_dict(state_dict, strict=strict)
    return model
        

def load_custom_pretrained(model, url, progress=False, check_hash=False):
    r"""Loads a custom (read non .pth) weight file
    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.
    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.
    Args:
        model: The instantiated model to load weights into
        progress (bool, optional): whether or not to display a progress bar to stderr. Default: False
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file. Default: False
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        _logger.warning('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    model.load_pretrained(cached_file)