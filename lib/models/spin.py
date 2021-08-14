"""
This script is brought from https://github.com/nkolot/SPIN
Adhere to their licence to use this script
"""

import math
import torch
import numpy as np
import os.path as osp
import torch.nn as nn

from lib.core.config import DATA_DIR
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J17, SMPL_MEAN_PARAMS


class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, feat_dim=2048, hidden_dim=1024, **kwargs):
        super(Regressor, self).__init__()

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            create_transl=False,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
        )
        npose = 24 * 6
        nshape = 10

        self.fc1 = nn.Linear(feat_dim + npose + nshape + 3, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hidden_dim, npose)
        self.decshape = nn.Linear(hidden_dim, nshape)
        self.deccam = nn.Linear(hidden_dim, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def iterative_regress(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        nt = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(nt, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(nt, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(nt, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        return pred_pose, pred_shape, pred_cam

    def forward(self, x, seqlen, J_regressor=None,
        init_pose=None, init_shape=None, init_cam=None, n_iter=3, **kwargs):
        nt = x.shape[0]
        N = nt//seqlen

        pred_pose, pred_shape, pred_cam = self.iterative_regress(x, init_pose, init_shape, init_cam, n_iter=3)
        output_regress = self.get_output(pred_pose, pred_shape, pred_cam, J_regressor)

        return output_regress


    def get_output(self, pred_pose, pred_shape, pred_cam, J_regressor):
        output = {}
        nt = pred_pose.shape[0]
        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(nt, -1, 3, 3)
        
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_vertices = pred_output.vertices[:nt]
        pred_joints = pred_output.joints[:nt]
        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(nt, -1)
        output['theta'] = torch.cat([pred_cam, pose, pred_shape], dim=1)
        output['verts'] = pred_vertices
        output['kp_2d'] = pred_keypoints_2d
        output['kp_3d'] = pred_joints
        output['rotmat'] = pred_rotmat
        return output


def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]
