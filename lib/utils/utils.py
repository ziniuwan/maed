# -*- coding: utf-8 -*-
"""
This script is brought from https://github.com/mkocabas/VIBE
Adhere to their licence to use this script
"""

import os
import yaml
import time
import torch
import shutil
import logging
import operator
import torch
from tqdm import tqdm
from os import path as osp
from functools import reduce
from typing import List, Union


def move_dict_to_device(dic, device, tensor2float=False):
    for k,v in dic.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dic[k] = v.float().to(device)
            else:
                dic[k] = v.to(device)
        elif isinstance(v, dict):
            move_dict_to_device(v, device)


def get_from_dict(dict, keys):
    return reduce(operator.getitem, keys, dict)


def tqdm_enumerate(iter, desc=""):
    i = 0
    for y in tqdm(iter, desc=desc):
        yield i, y
        i += 1


def iterdict(d):
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = dict(v)
            iterdict(v)
    return d


def accuracy(output, target):
    _, pred = output.topk(1)
    pred = pred.view(-1)

    correct = pred.eq(target).sum()

    return correct.item(), target.size(0) - correct.item()


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def step_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def read_yaml(filename):
    return yaml.load(open(filename, 'r'))


def write_yaml(filename, object):
    with open(filename, 'w') as f:
        yaml.dump(object, f)


def save_cfgnode_to_yaml(cfgnode, filename, mode='w'):
    with open(filename, mode) as f:
        f.write(cfgnode.dump())


def save_to_file(obj, filename, mode='w'):
    with open(filename, mode) as f:
        f.write(obj)


def concatenate_dicts(dict_list, dim=0):
    rdict = dict.fromkeys(dict_list[0].keys())
    for k in rdict.keys():
        rdict[k] = torch.cat([d[k] for d in dict_list], dim=dim)
    return rdict


def bool_to_string(x: Union[List[bool],bool]) ->  Union[List[str],str]:
    """
    boolean to string conversion
    :param x: list or bool to be converted
    :return: string converted thing
    """
    if isinstance(x, bool):
        return [str(x)]
    for i, j in enumerate(x):
        x[i]=str(j)
    return x


def checkpoint2model(checkpoint, key='gen_state_dict'):
    state_dict = checkpoint[key]
    print(f'Performance of loaded model on 3DPW is {checkpoint["performance"]:.2f}mm')
    # del state_dict['regressor.mean_theta']
    return state_dict


def get_optimizer(model, optim_type, lr, weight_decay, momentum):
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(
            lr=lr, 
            params=[{'params': p, 'name': n} for n, p in model.named_parameters()], 
            momentum=momentum
        )
    elif optim_type in ['Adam', 'adam', 'ADAM']:
        opt = torch.optim.Adam(
            lr=lr, 
            params=[{'params': p, 'name': n} for n, p in model.named_parameters()], 
            weight_decay=weight_decay
        )
    else:
        raise ModuleNotFoundError
    return opt


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{cfg.EXP_NAME}'

    logdir = osp.join(cfg.OUTPUT_DIR, logdir)
    os.makedirs(logdir, exist_ok=True)

    cfg.LOGDIR = logdir
    #cfg.TRAIN.PRETRAINED = osp.join(logdir, "model_best.pth.tar")

    # save config
    save_cfgnode_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg

def determine_output_feature_dim(inp_size, model):
    with torch.no_grad():
        # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
        # map for all networks, the feature metadata has reliable channel and stride info, but using
        # stride to calc feature dim requires info about padding of each stage that isn't captured.
        training = model.training
        if training:
            model.eval()
        o = model(torch.zeros(inp_size))
        if isinstance(o, (list, tuple)):
            o = o[-1]  # last feature if backbone outputs list/tuple of features
        feature_size = o.shape[-2:]
        feature_dim = o.shape[1]
        model.train(training)
    return feature_size, feature_dim