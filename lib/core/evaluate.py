import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
import traceback
import joblib
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d

from lib.core.config import DATA_DIR, DB_DIR
from lib.models.smpl import REGRESSOR_DICT, JID_DICT
from lib.utils.utils import move_dict_to_device, AverageMeter

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)
logger = logging.getLogger(__name__)

class Evaluator():
    def __init__(self):
        self.evaluation_accumulators = defaultdict(list)

    def inference(self, 
        model, 
        dataloader, 
        seqlen=8, 
        interp=1,
        device='cpu', 
        verbose=True, desc='[Evaluating] '
        ):
        """
        Args:
        interp (int >= 1): 1 out of <interp> frame is predicted by the model, while the rest is obtained by interpolation. interp = 1 means all the frames are predicted by the model. 
        """
        model.eval()
        dataset_name = dataloader.dataset.dataset_name

        start = time.time()

        summary_string = ''

        self.evaluation_accumulators = defaultdict(list)

        flatten_dim = lambda x: x.reshape((-1, ) + x.shape[2:])
        
        J_regressor = torch.from_numpy(np.load(osp.join(DATA_DIR, REGRESSOR_DICT[dataset_name]))).float() if REGRESSOR_DICT[dataset_name] else None
        Jid = JID_DICT[dataset_name]

        tqdm_bar = tqdm(range(len(dataloader)), desc=desc) if verbose else range(len(dataloader))
        test_iter = iter(dataloader)

        for i in tqdm_bar:
            target = next(test_iter)
            move_dict_to_device(target, device)

            # <=============
            with torch.no_grad():
                pred_verts_seq = []
                pred_j3d_seq = []
                pred_j2d_seq = []
                pred_theta_seq = []
                pred_rotmat_seq = []
                valid_joints = [joint_id for joint_id in range(target['kp_3d'].shape[2]) if target['kp_3d'][0,0,joint_id,-1]]

                orig_len = target['images'].shape[1]
                interp_len = target['images'][:, ::interp].shape[1]
                sample_freq = interp_len // seqlen
                
                for i in range(sample_freq):
                    inp = target['images'][:, ::interp][:, i::sample_freq]

                    preds = model(inp, J_regressor=J_regressor)

                    pred_verts_seq.append(preds['verts'].cpu().numpy())
                    pred_j3d_seq.append(preds['kp_3d'][:,:,Jid].cpu().numpy())
                    pred_j2d_seq.append(preds['kp_2d'][:,:,Jid].cpu().numpy())
                    pred_theta_seq.append(preds['theta'].cpu().numpy())
                    pred_rotmat_seq.append(preds['rotmat'].cpu().numpy())
                
                # valid_seq is used to filter out repeated frames
                valid_seq = flatten_dim(target['valid']).cpu().numpy()

                # register pred
                pred_verts_seq = self.interpolate(self.merge_sequence(pred_verts_seq), orig_len, interp_len)[valid_seq] # (NT, 6890, 3)
                pred_j3d_seq = self.interpolate(self.merge_sequence(pred_j3d_seq), orig_len, interp_len)[valid_seq] # (NT, n_kp, 3)
                pred_j2d_seq = self.interpolate(self.merge_sequence(pred_j2d_seq), orig_len, interp_len)[valid_seq] # (NT, n_kp, 2)
                pred_theta_seq = self.interpolate(self.merge_sequence(pred_theta_seq), orig_len, interp_len)[valid_seq] # (NT, 3+72+10)
                pred_rotmat_seq = self.interpolate(self.merge_sequence(pred_rotmat_seq), orig_len, interp_len)[valid_seq] # (NT, 3, 3)

                self.evaluation_accumulators['pred_verts'].append(pred_verts_seq)
                self.evaluation_accumulators['pred_theta'].append(pred_theta_seq)
                self.evaluation_accumulators['pred_rotmat'].append(pred_rotmat_seq)
                self.evaluation_accumulators['pred_j3d'].append(pred_j3d_seq)
                self.evaluation_accumulators['pred_j2d'].append(pred_j2d_seq)

                # register target
                target_j3d_seq = flatten_dim(target['kp_3d'][:, :, valid_joints]).cpu().numpy()[valid_seq] # (NT, n_kp, 4)
                target_j2d_seq = flatten_dim(target['kp_2d'][:, :, valid_joints]).cpu().numpy()[valid_seq] # (NT, n_kp, 3)
                target_theta_seq = flatten_dim(target['theta']).cpu().numpy()[valid_seq] # (NT, 3+72+10)
                self.evaluation_accumulators['target_theta'].append(target_theta_seq)
                self.evaluation_accumulators['target_j3d'].append(target_j3d_seq)
                self.evaluation_accumulators['target_j2d'].append(target_j2d_seq)

                # register some other infomation
                vid_name = np.reshape(np.array(target['instance_id']).T, (-1,))[valid_seq] # (NT,)
                paths = np.reshape(np.array(target['paths']).T, (-1,))[valid_seq] # (NT,)
                bboxes = np.reshape(target['bbox'].cpu().numpy(), (-1,4))[valid_seq] # (NT, 4)
                self.evaluation_accumulators['instance_id'].append(vid_name)
                self.evaluation_accumulators['bboxes'].append(bboxes)
                self.evaluation_accumulators['paths'].append(paths)

            # =============>

            batch_time = time.time() - start

            summary_string = f'{desc} | batch: {batch_time * 10.0:.4}ms '
            
            if verbose:
                tqdm_bar.set_description(summary_string)
    
    def merge_sequence(self, seq):
        if seq is None:
            return None 
        seq = np.stack(seq, axis=2) #(N, T//num_of_seq, num_of_seq, ...)
        assert len(seq.shape) >= 3
        seq = seq.reshape((-1, ) + seq.shape[3:]) #(NT, ...)
        return seq 

    def evaluate(self, save_path=''):
        # stack accumulators along axis 0
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.concatenate(v, axis=0)

        pred_j3ds = self.evaluation_accumulators['pred_j3d'] #(N, n_kp, 3)
        target_j3ds = self.evaluation_accumulators['target_j3d'][:,:,:-1] #(N, n_kp, 3)
        vis = self.evaluation_accumulators['target_j3d'][:,:,-1:] #(N, n_kp, 1)
        num_pred = len(pred_j3ds)
        target_j3ds *= vis
        pred_j3ds *= vis

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']
        pve = compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)
        accel_err = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
        accel = compute_accel(pred_j3ds)

        m2mm = 1000

        eval_dict = {
            'mpjpe': np.mean(errors) * m2mm,
            'pa-mpjpe': np.mean(errors_pa) * m2mm,
            'pve': np.mean(pve) * m2mm,
            'accel': np.mean(accel) * m2mm,
            'accel_err': np.mean(accel_err) * m2mm
        }
        
        if save_path:
            self.save_result(save_path, mpjpe=errors, pa_mpjpe=errors_pa, accel=accel_err)

        return eval_dict, num_pred
    
    def log(self, eval_dict, num_pred, desc=''):
        print(f"Evaluated on {int(num_pred)} number of poses.")
        print(f'{desc}' + ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()]))
    
    def run(self, model, dataloader, 
        seqlen=8, interp=1, device='cpu', 
        save_path='', verbose=True, desc='[Evaluating]'
        ):
        self.inference(model, dataloader, seqlen=seqlen, interp=interp, device=device, verbose=verbose, desc=desc)
        #self.count_attn(model)
        eval_dict, num_pred = self.evaluate(save_path)
        self.log(eval_dict, num_pred)

    def count_attn(self, model):
        result = {}
        result["vid_name"] = np.concatenate(self.evaluation_accumulators['instance_id'], axis=0)
        
        for i, blk in enumerate(model.backbone.blocks):
            result[f"attn_s_{i}"] = blk.attn.attn_count_s
            result[f"attn_t_{i}"] = blk.attn.attn_count_t
        
        joblib.dump(result, "attn.pt")
    
    def save_result(self, save_path, *args, **kwargs):
        save_fields = [
            'pred_theta', 
            #'pred_j3d', 
            #'pred_j2d', 
            'pred_verts', 
            'paths', 
            'bboxes', 
            #'pred_rotmat'
        ]
        save_dic = {k: v for k, v in self.evaluation_accumulators.items() if k in save_fields}
        save_dic.update(kwargs)
        joblib.dump(save_dic, osp.join(save_path, 'inference.pkl'))

    def interpolate(self, sequence, orig_len, interp_len):
        """
        Args:
        sequence (np array): size (N*interp_len, ...)
        orig_len (int)
        interp_len (int): larger than or equal to orig_len

        Return:
        A np array of size (N*orig_len, ...)
        """
        if orig_len == interp_len: return sequence
        sequence = sequence.reshape((-1, interp_len) + sequence.shape[1:]) # (N, interp_len, ...)
        x = np.linspace(1., 0., num=interp_len, endpoint=False)[::-1] # (interp_len, )
        f = interp1d(x, sequence, axis=1, fill_value="extrapolate")

        new_x = np.linspace(0., 1., num=orig_len, endpoint=True) # (orig_len, )
        ret = f(new_x)
        ret = ret.reshape((-1,) + ret.shape[2:])
        return ret
