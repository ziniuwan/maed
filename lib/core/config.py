# -*- coding: utf-8 -*-
import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
DB_DIR = 'data/database'
DATA_DIR = 'data/smpl_data'
INSTA_DIR = 'data/insta_variety'
INSTA_IMG_DIR = 'data/insta_variety_img'
MPII3D_DIR = 'data/mpi_inf_3dhp'
THREEDPW_DIR = 'data/3dpw'
HUMAN36M_DIR = 'data/human3.6m'
PENNACTION_DIR = 'data/penn_action'
POSETRACK_DIR = 'data/posetrack'

# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = True
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 8
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = -1
cfg.SAVE_FREQ = 5

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_2D = ['insta']
cfg.TRAIN.DATASETS_3D = ['mpii3d']
cfg.TRAIN.DATASETS_IMG = ['coco2014-all']
cfg.TRAIN.DATASET_EVAL = 'ThreeDPW'
cfg.TRAIN.BATCH_SIZE_3D = 4
cfg.TRAIN.BATCH_SIZE_2D = 4
cfg.TRAIN.BATCH_SIZE_IMG = 8
cfg.TRAIN.IMG_USE_FREQ = 1
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.RESUME = ''
cfg.TRAIN.NUM_ITERS_PER_EPOCH = -1

# <====== optimizer
cfg.TRAIN.OPTIM = CN()
cfg.TRAIN.OPTIM.OPTIM = 'Adam'
cfg.TRAIN.OPTIM.LR = 1e-4
cfg.TRAIN.OPTIM.WD = 1e-4
cfg.TRAIN.OPTIM.MOMENTUM = 0.9
cfg.TRAIN.OPTIM.WARMUP_EPOCH = 2
cfg.TRAIN.OPTIM.WARMUP_FACTOR = 0.1
cfg.TRAIN.OPTIM.MILESTONES = [10,15]

cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 20
cfg.DATASET.OVERLAP = 0.5
cfg.DATASET.SAMPLE_POOL = 64
cfg.DATASET.SIZE_JITTER = 0.2
cfg.DATASET.ROT_JITTER = 30
cfg.DATASET.RANDOM_SAMPLE = True
cfg.DATASET.RANDOM_START = False
cfg.DATASET.RANDOM_FLIP = 0.5
cfg.DATASET.WIDTH = 224
cfg.DATASET.HEIGHT = 224
cfg.DATASET.RANDOM_CROP_P = 0.0
cfg.DATASET.RANDOM_CROP_SIZE = 0.5
cfg.DATASET.COLOR_JITTER = 0.3
cfg.DATASET.ERASE_PROB = 0.3
cfg.DATASET.ERASE_PART = 0.7
cfg.DATASET.ERASE_FILL = False
cfg.DATASET.ERASE_KP = False
cfg.DATASET.ERASE_MARGIN = 0.2


cfg.LOSS = CN()
cfg.LOSS.KP_2D_W = 60.
cfg.LOSS.KP_3D_W = 30.
cfg.LOSS.SHAPE_W = 0.001
cfg.LOSS.POSE_W = 1.0
cfg.LOSS.SMPL_NORM = 1.
cfg.LOSS.ACCL_W = 0.

cfg.MODEL = CN()

# GRU model hyperparams
cfg.MODEL.DECODER = CN()
cfg.MODEL.DECODER.BACKBONE = 'ktd'
cfg.MODEL.DECODER.HIDDEN_DIM = 1024
cfg.MODEL.ENCODER = CN()
cfg.MODEL.ENCODER.BACKBONE = 'ste'
cfg.MODEL.ENCODER.NUM_BLOCKS = 6
cfg.MODEL.ENCODER.NUM_HEADS = 12
cfg.MODEL.ENCODER.SPA_TEMP_MODE = 'vanilla'


cfg.EVAL = CN()
cfg.EVAL.SEQLEN = 16
cfg.EVAL.SAMPLE_POOL = 128
cfg.EVAL.BATCH_SIZE = 32
cfg.EVAL.INTERPOLATION = 1

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--pretrained', type=str, help='stage 1 checkpoint file path', default='')
    parser.add_argument('--eval_ds', type=str, help='eval set name', default='3dpw')
    parser.add_argument('--eval_set', type=str, help='eval set in [test|val]', default='test')
    parser.add_argument('--image_root', type=str, help='inference image root', default='')
    parser.add_argument('--image_list', type=str, help='inference image list', default='')
    parser.add_argument('--output_path', type=str, help='path to save the inference file generated in evaluation', default='')
    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return args, cfg, cfg_file
