from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from lib.dataset import *
import torch
import joblib
import os.path as osp

def get_data_loaders(cfg, transforms_3d, transforms_2d, transforms_val, transforms_img, rank, world_size, verbose=True):
    def get_2d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = VideoDataset(
                dataset_name=dataset_name,
                set='train', 
                transforms=transforms_2d,
                seqlen=cfg.DATASET.SEQLEN, 
                overlap=cfg.DATASET.OVERLAP, 
                sample_pool=cfg.DATASET.SAMPLE_POOL,
                random_sample=cfg.DATASET.RANDOM_SAMPLE,
                random_start=cfg.DATASET.RANDOM_START,
                verbose=verbose,
                debug=cfg.DEBUG
                )
            datasets.append(db)
        return ConcatDataset(datasets)

    def get_3d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = VideoDataset(
                dataset_name=dataset_name,
                set='train', 
                transforms=transforms_3d,
                seqlen=cfg.DATASET.SEQLEN, 
                overlap=cfg.DATASET.OVERLAP if dataset_name != '3dpw' else 8, 
                sample_pool=cfg.DATASET.SAMPLE_POOL,
                random_sample=cfg.DATASET.RANDOM_SAMPLE,
                random_start=cfg.DATASET.RANDOM_START,
                verbose=verbose,
                debug=cfg.DEBUG, 
                )
            datasets.append(db)
        return ConcatDataset(datasets)

    def get_img_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = ImageDataset(
                dataset_name=dataset_name,
                set='train', 
                transforms=transforms_img,
                verbose=verbose,
                debug=cfg.DEBUG, 
                )
            if dataset_name == 'mpii3d':
                db = Subset(db, list(range(len(db)))[::5])
            datasets.append(db)
        return ConcatDataset(datasets)

    # ===== Video 2D keypoints datasets =====
    train_2d_dataset_names = cfg.TRAIN.DATASETS_2D
    data_2d_batch_size = cfg.TRAIN.BATCH_SIZE_2D

    if data_2d_batch_size:
        train_2d_db = get_2d_datasets(train_2d_dataset_names)
        train_2d_sampler = DistributedSampler(train_2d_db, rank=rank, num_replicas=world_size)
        train_2d_loader = DataLoader(
            dataset=train_2d_db,
            batch_size=data_2d_batch_size,
            #shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            sampler=train_2d_sampler
        )
    else:
        train_2d_loader = None

    # ===== Video 3D keypoint datasets =====
    train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
    data_3d_batch_size = cfg.TRAIN.BATCH_SIZE_3D

    if data_3d_batch_size:
        train_3d_db = get_3d_datasets(train_3d_dataset_names)
        train_3d_sampler = DistributedSampler(train_3d_db, rank=rank, num_replicas=world_size)
        train_3d_loader = DataLoader(
            dataset=train_3d_db,
            batch_size=data_3d_batch_size,
            #shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            sampler=train_3d_sampler
        )
    else:
        train_3d_loader = None
    
    # ===== Image datasets =====
    train_img_dataset_names = cfg.TRAIN.DATASETS_IMG
    data_img_batch_size = cfg.TRAIN.BATCH_SIZE_IMG

    if data_img_batch_size:
        train_img_db = get_img_datasets(train_img_dataset_names)
        train_img_sampler = DistributedSampler(train_img_db, rank=rank, num_replicas=world_size)
        train_img_loader = DataLoader(
            dataset=train_img_db,
            batch_size=data_img_batch_size,
            num_workers=cfg.NUM_WORKERS,
            sampler=train_img_sampler,
        )
    else:
        train_img_loader = None

    # ===== Evaluation dataset =====
    valid_db = VideoDataset(
        dataset_name=cfg.TRAIN.DATASET_EVAL,
        set='val', 
        transforms=transforms_val,
        overlap=0, 
        sample_pool=cfg.EVAL.SAMPLE_POOL,
        random_sample=False,
        random_start=False,
        verbose=verbose,
        debug=cfg.DEBUG
        )
    valid_sampler = DistributedSampler(valid_db, rank=rank, num_replicas=world_size)

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.EVAL.BATCH_SIZE,
        #shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        sampler=valid_sampler
    )

    return train_2d_loader, train_3d_loader, valid_loader, train_img_loader
