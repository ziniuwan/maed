import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.models.spin import projection
from lib.utils.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

ANCESTOR_INDEX = [
    [],
    [0], 
    [0], 
    [0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 1, 4],
    [0, 2, 5],
    [0, 3, 6],
    [0, 1, 4, 7],
    [0, 2, 5, 8],
    [0, 3, 6, 9], 
    [0, 3, 6, 9], 
    [0, 3, 6, 9],
    [0, 3, 6, 9, 12],
    [0, 3, 6, 9, 13],
    [0, 3, 6, 9, 14],
    [0, 3, 6, 9, 13, 16],
    [0, 3, 6, 9, 14, 17],
    [0, 3, 6, 9, 13, 16, 18],
    [0, 3, 6, 9, 14, 17, 19],
    [0, 3, 6, 9, 13, 16, 18, 20],
    [0, 3, 6, 9, 14, 17, 19, 21]
]

class KTD(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=1024, **kwargs):
        super(KTD, self).__init__()

        self.feat_dim = feat_dim
        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            create_transl=False,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
        )
        npose_per_joint = 6
        nshape = 10
        ncam = 3

        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        
        self.joint_regs = nn.ModuleList()
        for joint_idx, ancestor_idx in enumerate(ANCESTOR_INDEX):
            regressor = nn.Linear(hidden_dim + npose_per_joint * len(ancestor_idx), npose_per_joint)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)

        self.decshape = nn.Linear(hidden_dim, nshape)
        self.deccam = nn.Linear(hidden_dim, ncam)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x, seqlen, J_regressor=None, 
        return_shape_cam=False, **kwargs):
        nt = x.shape[0]
        N = nt//seqlen

        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        pred_shape = self.decshape(x)
        pred_cam = self.deccam(x)
        
        pose = []
        for ancestor_idx, reg in zip(ANCESTOR_INDEX, self.joint_regs):
            ances = torch.cat([x] + [pose[i] for i in ancestor_idx], dim=1)
            pose.append(reg(ances))

        pred_pose = torch.cat(pose, dim=1)

        if return_shape_cam:
            return pred_shape, pred_cam
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
