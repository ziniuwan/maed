# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from lib.utils.geometry import batch_rodrigues

class _LossBase(nn.Module):
    def __init__(
            self,
            device='cuda',
    ):
        super(_LossBase, self).__init__()

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_accl = nn.MSELoss().to(self.device)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        Args:
            pred_keypoints_2d: tensor of shape [N, T, n_kp, 2] or [N, n_kp, 2]. n_kp is typically 49.
            gt_keypoints_2d: tensor of shape [N, T, n_kp, 3] or [N, n_kp, 3]. n_kp is typically 49.
        """
        if len(gt_keypoints_2d) > 0:
            if len(gt_keypoints_2d.shape) > 3: # video input
                gt_keypoints_2d = gt_keypoints_2d.reshape((-1,) + gt_keypoints_2d.shape[2:])
                pred_keypoints_2d = pred_keypoints_2d.reshape((-1,) + pred_keypoints_2d.shape[2:])

            conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
            loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean() #.sum() / conf.sum()
            return loss
        else:
            return torch.tensor(0).fill_(0.).float().to(self.device)

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        Args:
            pred_keypoints_3d: tensor of shape [N, T, n_kp, 3] or [N, n_kp, 3]. n_kp is typically 14.
            gt_keypoints_3d: tensor of shape [N, T, n_kp, 4] or [N, n_kp, 4]. n_kp is typically 14.
        """
        if len(gt_keypoints_3d) > 0:
            if len(gt_keypoints_3d.shape) > 3: # video input
                gt_keypoints_3d = gt_keypoints_3d.reshape((-1,) + gt_keypoints_3d.shape[2:])
                pred_keypoints_3d = pred_keypoints_3d.reshape((-1,) + pred_keypoints_3d.shape[2:])

            conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
            gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]

            gt_pelvis = (gt_keypoints_3d[:, 25+2,:] + gt_keypoints_3d[:, 25+3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 25+2,:] + pred_keypoints_3d[:, 25+3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.tensor(0).fill_(0.).float().to(self.device)

    def smpl_losses(self, pred_pose, pred_shape, gt_pose, gt_shape, w_smpl):
        """
        Compute SMPL parameters loss for the examples that SMPL annotations are available.
        Args:
            pred_pose: tensor of shape [N, T, 72] or [N, 72].
            pred_shape: tensor of shape [N, T, 10] or [N, 10].
            gt_pose: tensor of the same shape as pred_pose.
            gt_shape: tensor of the same shape as pred_shape.
            w_smpl: bool tensor of shape [N, T] or [N, ]
        """
        if len(pred_pose.shape) > 2: # video input
            w_smpl = w_smpl.reshape(-1)
            pred_pose = pred_pose.reshape((-1,) + pred_pose.shape[2:])[w_smpl]
            pred_shape = pred_shape.reshape((-1,) + pred_shape.shape[2:])[w_smpl]
            gt_pose = gt_pose.reshape((-1,) + gt_pose.shape[2:])[w_smpl]
            gt_shape = gt_shape.reshape((-1,) + gt_shape.shape[2:])[w_smpl]

        if len(pred_pose) > 0:
            pred_rotmat_valid = batch_rodrigues(pred_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
            gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
            pred_shape_valid = pred_shape
            gt_shape_valid = gt_shape
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_shape = self.criterion_regr(pred_shape_valid, gt_shape_valid)
        else:
            loss_regr_pose = torch.tensor(0).fill_(0.).float().to(self.device)
            loss_regr_shape = torch.tensor(0).fill_(0.).float().to(self.device)
        return loss_regr_pose, loss_regr_shape
    
    def accl_losses(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute joints3d accel loss.
        Args:
            pred_keypoints_3d: tensor of shape [N, T, n_kp, 3]. n_kp is typically 49.
            gt_keypoints_3d: tensor of shape [N, T, n_kp, 4] or [N, n_kp, 4]. n_kp is typically 49.
        """
        if len(pred_keypoints_3d) > 0:
            conf = gt_keypoints_3d[:, :, :, -1].unsqueeze(-1) #(N, T, 49, 1)
            conf_velocity = conf[:, 1:] * conf[:, 1:] #(N, T-1, 49, 1)
            conf_accl = conf_velocity[:, 1:] * conf_velocity[:, 1:] #(N, T-2, 49, 1)

            pred_velocity = pred_keypoints_3d[:, 1:] - pred_keypoints_3d[:, :-1] #(N, T-1, 49, 3)
            pred_accl = pred_velocity[:, 1:] - pred_velocity[:, :-1] #(N, T-2, 49, 3)
            pred_accl = pred_accl * conf_accl #(N, T-2, 49, 3)
            
            gt_keypoints_3d = gt_keypoints_3d[:, :, :, :3] #(N, T, 49, 3)
            gt_velocity = gt_keypoints_3d[:, 1:] - gt_keypoints_3d[:, :-1] #(N, T-1, 49, 3)
            gt_accl = gt_velocity[:, 1:] - gt_velocity[:, :-1] #(N, T-2, 49, 3)
            gt_accl = gt_accl * conf_accl #(N, T-2, 49, 3)

            loss_accl = self.criterion_accl(pred_accl, gt_accl)
            return loss_accl
        else:
            return torch.tensor(0).fill_(0.).float().to(self.device)
    


class LossVideo(_LossBase):
    """
    Compute loss between VIDEO predictions and labels.
    Input:
        preds: A dict object containing:
            preds["kp_2d"]: tensor of shape [N, T, n_kp, 3], n_kp is typically 49.
            preds["kp_3d"]: tensor of shape [N, T, n_kp, 4], n_kp is typically 49.
            preds["theta"]: tensor of shape [N, T, 3+72+10]. 3 for cam, 72 for SMPL pose, 10 for SMPL shape.
            preds["kp_3d_branch2"] (optional): tensor of shape [N, T1, n_kp, 4], where typically T1 > T.

        data_3d: A dict object containing: 
            data_3d["kp_2d"], data_3d["kp_3d"], data_3d["theta"]: tensor of the same shape as preds.
            data_3d["w_smpl"]: tensor of shape [N, T], indicating whether the corresponding frames' labels are valid.
            data_3d["kp_3d_full"] (optional): tensor of the same shape as preds["kp_3d_branch2"].

        data_2d (optional): A dict object containing:
            data_2d["kp_2d"]: the same as preds.
    """
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=30.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            e_smpl_norm_loss = 1.,
            e_smpl_accl_loss = 0.,
            device='cuda',
    ):
        super(LossVideo, self).__init__(device)
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.e_smpl_norm_loss = e_smpl_norm_loss
        self.e_smpl_accl_loss = e_smpl_accl_loss
        self.e_3d_loss_weight_branch2 = 300.
        self.e_loss_weight_branch2 = 300.

    def forward(
            self,
            preds,
            data_3d,
            data_2d,
    ):

        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            gt_j2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        else:
            sample_2d_count = 0
            gt_j2d = data_3d['kp_2d']

        gt_j3d = data_3d['kp_3d']
        data_3d_theta = data_3d['theta']
        
        w_smpl = data_3d['w_smpl'].type(torch.bool)

        pred_j2d = preds['kp_2d']
        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]

        # <======== Generator Loss
        loss_kp_2d = self.e_loss_weight * self.keypoint_loss(pred_j2d, gt_j2d)
        loss_kp_3d = self.e_3d_loss_weight * self.keypoint_3d_loss(pred_j3d, gt_j3d)

        loss_dict = {
            'loss_kp_2d': loss_kp_2d,
            'loss_kp_3d': loss_kp_3d,
        }

        real_shape, pred_shape = data_3d_theta[:, :, 75:], pred_theta[:, :, 75:]
        real_pose, pred_pose = data_3d_theta[:, :, 3:75], pred_theta[:, :, 3:75]

        if self.e_shape_loss_weight > 0 and self.e_pose_loss_weight > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape, w_smpl)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose

        if self.e_smpl_norm_loss > 0:
            nt = pred_theta.shape[0] * pred_theta.shape[1]
            loss_dict['loss_norm'] = self.e_smpl_norm_loss * torch.norm(pred_theta.reshape((-1,) + pred_theta.shape[2:])[:, 3:], p=2, dim=(0,1)) / nt

        if self.e_smpl_accl_loss > 0:
            loss_dict['loss_accl'] = self.e_smpl_accl_loss * self.accl_losses(pred_j3d, gt_j3d)

        total_loss = torch.stack(list(loss_dict.values())).sum()

        return total_loss, loss_dict   



class LossImage(_LossBase):
    """
    Compute loss between IMAGE predictions and labels.
    Input:
        preds: A dict object containing:
            preds["kp_2d"]: tensor of shape [N, n_kp, 3], n_kp is typically 49.
            preds["theta"]: tensor of shape [N, 3+72+10]. 3 for cam, 72 for SMPL pose, 10 for SMPL shape.

        target: A dict object containing: 
            target["kp_2d"], target["theta"]: the same as preds.
            target["w_smpl"]: tensor of shape [N,], indicating whether the corresponding images' labels are valid.
    """
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=600.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            e_smpl_norm_loss = 1.,
            device='cuda',
    ):
        super(LossImage, self).__init__(device)
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.e_smpl_norm_loss = e_smpl_norm_loss
        self.e_loss_weight_branch2 = 300.

    def forward(
            self,
            preds,
            target,
    ):
        gt_j2d = target['kp_2d']
        gt_j3d = target['kp_3d'] if 'kp_3d' in target else None
        real_theta = target['theta']

        pred_j2d = preds['kp_2d'].squeeze(1)
        pred_j3d = preds['kp_3d'].squeeze(1)
        pred_theta = preds['theta'].squeeze(1)

        w_smpl = target['w_smpl'].type(torch.bool)

        # <======== Generator Loss
        loss_kp_2d =  self.keypoint_loss(pred_j2d, gt_j2d) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, gt_j3d) * self.e_3d_loss_weight if gt_j3d is not None else 0.

        real_shape, pred_shape = real_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = real_theta[:, 3:75], pred_theta[:, 3:75]

        loss_dict = {
            'loss_kp_2d': loss_kp_2d,
            'loss_kp_3d': loss_kp_3d,
        }
        if self.e_shape_loss_weight > 0 and self.e_pose_loss_weight > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape, w_smpl)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose

        if self.e_smpl_norm_loss > 0:
            nt = pred_theta.shape[0]
            loss_dict['loss_norm'] = self.e_smpl_norm_loss * torch.norm(pred_theta[:, 3:], p=2, dim=(0,1)) / nt

        total_loss = torch.stack(list(loss_dict.values())).sum()

        return total_loss, loss_dict

class Loss(nn.Module):
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=30.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            e_smpl_norm_loss = 1.,
            e_smpl_accl_loss = 0.,
            device='cuda',
    ):
        super(Loss, self).__init__()
        self.loss_video = LossVideo(
            e_loss_weight,
            e_3d_loss_weight,
            e_pose_loss_weight,
            e_shape_loss_weight,
            e_smpl_norm_loss,
            e_smpl_accl_loss,
            device
        )
        self.loss_image = LossImage(
            e_loss_weight,
            e_3d_loss_weight,
            e_pose_loss_weight,
            e_shape_loss_weight,
            e_smpl_norm_loss,
            device
        )
        
    def forward(self, preds, **kwargs):
        if 'target_2d' in kwargs:
            loss, loss_dict = self.loss_video(
                preds, 
                kwargs['target_3d'], 
                kwargs['target_2d'],
            )
        elif 'target_img' in kwargs:
            loss, loss_dict = self.loss_image(
                preds, 
                kwargs['target_img'], 
            )
        else:
            loss = 0
            loss_dict = {}
        
        return loss, loss_dict 
    
    def merge_loss(self, loss_vid, loss_vid_dict, loss_img, loss_img_dict, vid_w=1.0, img_w=1.0):
        loss_dict = {}
        keys = set(list(loss_vid_dict.keys()) + list(loss_img_dict.keys()))
        for k in keys:
            _loss = 0
            if k in loss_vid_dict:
                _loss = _loss + loss_vid_dict[k] * vid_w
            if k in loss_img_dict:
                _loss = _loss + loss_img_dict[k] * img_w
            loss_dict[k] = _loss

        loss = loss_vid * vid_w + loss_img * img_w

        return loss, loss_dict


def concat_mask(masks_2d, masks_3d):
    result = {}
    for k in masks_2d.keys():
        result[k] = torch.cat([masks_2d[k], masks_3d[k]], 0) #(concat along batch axis)
    return result


def batch_encoder_disc_l2_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_wasserstein_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return -1 * disc_value.sum() / k


def batch_adv_disc_wasserstein_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''

    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]

    la = -1 * real_disc_value.sum() / ka
    lb = fake_disc_value.sum() / kb
    return la, lb, la + lb


def batch_smooth_pose_loss(pred_theta):
    pose = pred_theta[:,:,3:75]
    pose_diff = pose[:,1:,:] - pose[:,:-1,:]
    return torch.mean(pose_diff).abs()


def batch_smooth_shape_loss(pred_theta):
    shape = pred_theta[:, :, 75:]
    shape_diff = shape[:, 1:, :] - shape[:, :-1, :]
    return torch.mean(shape_diff).abs()
