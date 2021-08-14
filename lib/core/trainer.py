# -*- coding: utf-8 -*-

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp

from tqdm import tqdm
from collections import defaultdict

from torch.distributed import all_reduce as allreduce

from lib.core.evaluate import Evaluator
from lib.utils.utils import move_dict_to_device, AverageMeter
from lib.models.smpl import J49_TO_J14, H36M_TO_J14

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(
            self,
            data_loaders,
            model,
            optimizer,
            end_epoch,
            criterion,
            start_epoch=0,
            img_use_freq=1,
            lr_scheduler=None,
            device=None,
            writer=None,
            debug=False,
            debug_freq=10,
            logdir='output',
            resume='',
            performance_type='min',
            seqlen=8,
            interp=1,
            rank=0,
            world_size=1,
            num_iters_per_epoch=1000,
            save_freq=5,
    ):

        # Prepare dataloaders
        self.train_2d_loader, self.train_3d_loader, self.valid_loader, self.train_img_loader = data_loaders

        self.train_2d_iter = self.train_3d_iter = self.train_img_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            self.train_3d_iter = iter(self.train_3d_loader)

        if self.train_img_loader:
            self.train_img_iter = iter(self.train_img_loader)


        # Models and optimizers
        self.model = model
        self.optimizer = optimizer

        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.img_use_freq = img_use_freq
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir

        self.performance_type = performance_type
        self.train_global_step = 0
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')
        self.save_freq = save_freq

        # determine iters per epoch
        if num_iters_per_epoch<=0:
            if self.train_3d_loader:
                self.num_iters_per_epoch = len(self.train_3d_loader)
            elif self.train_2d_loader:
                self.num_iters_per_epoch = len(self.train_2d_loader)
            else:
                self.num_iters_per_epoch = len(self.train_img_loader)
        else:
            self.num_iters_per_epoch = num_iters_per_epoch

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize Evaluator for validation
        self.evaluator = Evaluator()
        self.seqlen = seqlen
        self.interp = interp

        # Resume from a pretrained model
        if resume:
            self.resume_pretrained(resume)

    def train(self):
        # Single epoch training routine

        losses = defaultdict(lambda: AverageMeter())

        timer = {
            'data': 0,
            'forward': 0,
            'backward': 0,
            'batch': 0,
        }

        self.model.train()

        tqdm_bar = tqdm(range(self.num_iters_per_epoch), desc="[ Training ] ") if self.rank==0 else range(self.num_iters_per_epoch)
        for i in tqdm_bar:
            summary_string = ''

            # <======= Get training data
            start = time.time()
            # Dirty solution to reset an iterator
            target_2d = target_3d = target_img = None
            if self.train_2d_iter:
                try:
                    target_2d = next(self.train_2d_iter)
                except StopIteration:
                    self.train_2d_loader.sampler.set_epoch(self.epoch)
                    self.train_2d_iter = iter(self.train_2d_loader)
                    target_2d = next(self.train_2d_iter)

                move_dict_to_device(target_2d, self.device)

            if self.train_3d_iter:
                try:
                    target_3d = next(self.train_3d_iter)
                except StopIteration:
                    self.train_3d_loader.sampler.set_epoch(self.epoch)
                    self.train_3d_iter = iter(self.train_3d_loader)
                    target_3d = next(self.train_3d_iter)

                move_dict_to_device(target_3d, self.device)
            
            if self.train_img_iter and (i+1) % self.img_use_freq==0:
                try:
                    target_img = next(self.train_img_iter)
                except StopIteration:
                    self.train_img_loader.sampler.set_epoch(self.epoch)
                    self.train_img_iter = iter(self.train_img_loader)
                    target_img = next(self.train_img_iter)

                move_dict_to_device(target_img, self.device)

            if target_2d and target_3d:
                inp_vid = torch.cat((target_2d['images'], target_3d['images']), dim=0).to(self.device)
            elif target_3d:
                inp_vid = target_3d['images'].to(self.device)
            elif target_2d:
                inp_vid = target_2d['images'].to(self.device)


            if target_img:
                inp_img = target_img['image'].to(self.device)
                inp_img = inp_img.unsqueeze(1)


            timer['data'] = time.time() - start
            # =======>

            # <======= Feedforward and backward
            if target_2d or target_3d:
                nt_vid = inp_vid.shape[0] * inp_vid.shape[1]
                loss_vid, loss_vid_dict, forward_vid_time = self.forward(inp_vid, target_3d=target_3d, target_2d=target_2d)
            else:
                nt_vid = backward_vid_time = forward_vid_time = loss_vid = 0
                loss_vid_dict = {}

            if target_img:
                nt_img = inp_img.shape[0]
                loss_img, loss_img_dict, forward_img_time = self.forward(inp_img, target_img=target_img)
            else:
                nt_img = backward_img_time = forward_img_time = loss_img = 0
                loss_img_dict = {}
            
            loss_weight_vid = nt_vid / (nt_img+nt_vid)
            loss_weight_img = 1 - loss_weight_vid
            timer['backward'] = self.backward(loss_img * loss_weight_img + loss_vid * loss_weight_vid)
            timer['forward'] = forward_img_time + forward_vid_time

            total_loss, loss_dict = self.criterion.merge_loss(loss_vid, loss_vid_dict, loss_img, loss_img_dict, vid_w=loss_weight_vid, img_w=loss_weight_img)
            # =======>

            # <======= Log training info
            total_loss, total_instance = self.sync_data(total_loss)

            self.loss_meter_update(losses, total_loss.item(), loss_dict, total_instance)

            timer['batch'] = timer['data'] + timer['forward'] + timer['backward']

            summary_string = f'[ Training ] epoch: ({self.epoch + 1}/{self.end_epoch})'

            for k, v in losses.items():
                summary_string += f' | {k}: {v.avg:.2f}'
                if self.writer:
                    self.writer.add_scalar('train_loss/'+k, v.avg, global_step=self.train_global_step)

            for k,v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            if self.writer:
                self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            if self.debug and self.train_global_step % self.debug_freq==0 and self.rank==0:
                import pdb; pdb.set_trace()

            if self.rank == 0:
                tqdm_bar.set_description(summary_string)

            self.train_global_step += 1

            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>
   
    def backward(self, loss):
        start = time.time()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        cost = time.time() - start
        return cost

    def forward(self, inp, **kwargs):
        start = time.time()

        preds = self.model(inp)
        loss, loss_dict = self.criterion(preds=preds, **kwargs)

        cost = time.time() - start
        return loss, loss_dict, cost 

    def sync_data(self, item, num_item=1):
        # Gather data computed on each rank and average them
        if not torch.is_tensor(item):
            item = torch.tensor(item*num_item).float().to(self.device)
            return_tensor = False
        else:
            item = item*num_item
            return_tensor = True

        if not torch.is_tensor(num_item):
            num_item = torch.tensor(num_item).float().to(self.device)
        allreduce(item) # Sum of data
        allreduce(num_item) # Total number of data     

        item_avg = item.item()
        num_total = num_item.item()
        item_avg /= max(num_total, 1)

        if return_tensor:
            item_avg = torch.tensor(item_avg).float().to(self.device)
        return item_avg, num_total
    
    def loss_meter_update(self, loss_meter, total_loss, loss_dict, num):
        loss_meter['loss'].update(total_loss, num)
        for key, loss in loss_dict.items():
            loss_meter[key].update(loss, num)
        return loss_meter

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            # train one epoch
            self.train()
            # validation
            if (self.epoch+1) % self.save_freq == 0:
                self.evaluator.inference(
                    model=self.model,
                    dataloader=self.valid_loader,
                    seqlen=self.seqlen,
                    interp=self.interp,
                    device=self.device,
                    verbose=self.rank==0,
                    desc=f'[Validation] epoch: ({self.epoch + 1}/{self.end_epoch})'
                )
                eval_dict, num_pred = self.evaluator.evaluate()
                # log validation info
                for k,v in eval_dict.items():
                    eval_dict[k], num_pred_all = self.sync_data(v, num_pred)
                if self.rank==0:
                    self.evaluator.log(eval_dict, num_pred_all)
                performance = eval_dict['pa-mpjpe']
                # tensorboard info
                if self.writer:
                    for k,v in eval_dict.items():
                        self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)
    
                if self.rank == 0:
                    logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')
                    self.save_model(performance, epoch+1)
            
            if self.writer:
                for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar('lr', param_group['lr'], global_step=self.epoch)
            # adjust learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

        if self.writer:
            self.writer.close()

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, 'epoch_{}.pth.tar'.format(epoch))
        torch.save(save_dict, filename)
        
        is_best = performance < self.best_performance if self.performance_type == 'min' else performance > self.best_performance

        if is_best:
            logger.info('Best performance achived, saving it to model_best.pth.tar!')

            filename = osp.join(self.logdir, 'model_best.pth.tar')
            self.best_performance = performance

            torch.save(save_dict, filename)
            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))


    def resume_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_performance = checkpoint['performance']
            if self.rank == 0:
                logger.info(f"=> loaded checkpoint '{model_path}' "
                      f"(epoch {self.start_epoch}, performance {self.best_performance})")
            
            del checkpoint
        else:
            if self.rank == 0:
                logger.info(f"=> no checkpoint found at '{model_path}'")
