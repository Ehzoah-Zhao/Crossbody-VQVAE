"""utils/physics_losses.py - FK-based physical consistency losses for 363-dim G1 data
363-dim layout (v3/v4): r_velocity(1)+root_vel_xz(2)+root_y(1)+ric(87)+rot(180)+local_vel(90)+contacts(2)
363-dim layout: r_velocity(1)+root_vel_xz(2)+root_y(1)+ric(87)+rot(180)+local_vel(90)+contacts(2)
303-dim layout (v5 quat): r_velocity(1)+root_vel_xz(2)+root_y(1)+ric(87)+quat(120)+local_vel(90)+contacts(2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

G1_363_SLICES = {
    'r_velocity':   (0, 1),
    'root_vel_xz':  (1, 3),
    'root_y':       (3, 4),
    'ric_data':     (4, 91),
    'rot_data':     (91, 271),
    'local_vel':    (271, 361),
    'contacts':     (361, 363),
}

# v5: 303-dim quaternion format (30 joints x 4 = 120D for rotations)
G1_303_SLICES = {
    'r_velocity':   (0, 1),
    'root_vel_xz':  (1, 3),
    'root_y':       (3, 4),
    'ric_data':     (4, 91),
    'quat_data':    (91, 211),   # 30 x 4 = 120D
    'local_vel':    (211, 301),
    'contacts':     (301, 303),
}
G1_N_JOINTS = 30
G1_CONTACT_JOINTS = [6, 12]  # left foot, right foot

G1_BONE_OFFSETS = [
    [ 0.0,       0.0,       0.0     ], [ -0.064452, -0.1027,    0.0     ],
    [ -0.052,    -0.030465,  0.0     ], [ 0.0,      -0.12412,   0.025001],
    [ -0.0021489,-0.17734,  -0.078273], [ 0.000094, -0.30001,   0.0     ],
    [ 0.0,      -0.017558,  0.0     ], [ 0.064452, -0.1027,    0.0     ],
    [ 0.052,    -0.030465,  0.0     ], [ 0.0,      -0.12412,   0.025001],
    [ 0.0021489,-0.17734,  -0.078273], [ -0.000094, -0.30001,   0.0     ],
    [ 0.0,      -0.017558,  0.0     ], [ 0.0,       0.0,       0.0     ],
    [ 0.0,       0.044,    -0.003964], [ 0.0,       0.0,       0.0     ],
    [ -0.10022,   0.24778,   0.003956], [ -0.038,    -0.013831,  0.0     ],
    [ -0.00624,  -0.1032,    0.0     ], [ 0.0,      -0.080518,  0.015783],
    [ -0.001888, -0.010,     0.100   ], [ 0.0,       0.0,       0.038   ],
    [ 0.0,       0.0,       0.046   ], [ 0.10021,   0.24778,   0.003956],
    [ 0.038,    -0.013831,  0.0     ], [ 0.00624,  -0.1032,    0.0     ],
    [ 0.0,      -0.080518,  0.015783], [ 0.001888, -0.010,     0.100   ],
    [ 0.0,       0.0,       0.038   ], [ 0.0,       0.0,       0.046   ],
]

G1_KINEMATIC_TREE = [
    [0, 1, 2, 3, 4, 5, 6],
    [0, 7, 8, 9, 10, 11, 12],
    [0, 13, 14, 15],
    [15, 16, 17, 18, 19, 20, 21, 22],
    [15, 23, 24, 25, 26, 27, 28, 29],
]


class FKPositionLoss(nn.Module):
    """FK-based joint position consistency loss.
    Losses: fk_sup(supervised) + fk_sc(self-consistent) + fk_ee(end-effector) + contact + bone_len
    """
    def __init__(self, supervised_weight=1.0, self_consist_weight=0.3, ee_weight=2.0):
        super().__init__()
        off = torch.tensor(G1_BONE_OFFSETS, dtype=torch.float32)
        self.register_buffer('offsets', off)
        self.tree = G1_KINEMATIC_TREE
        self.ee_idx = G1_CONTACT_JOINTS
        self.sw, self.cw, self.ew = supervised_weight, self_consist_weight, ee_weight

    @staticmethod
    def rot6d_to_mat(d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def fk(self, rot_6d):
        *bd, J, _ = rot_6d.shape
        off = self.offsets.to(rot_6d.device).view(*([1]*len(bd)), J, 3).expand(*bd, -1, -1)
        rm = self.rot6d_to_mat(rot_6d)
        jts = torch.zeros(*bd, J, 3, device=rot_6d.device)
        for chain in self.tree:
            Rg = rm[..., chain[0], :, :]
            for i in range(1, len(chain)):
                p, c = chain[i-1], chain[i]
                jts[..., c, :] = jts[..., p, :] + torch.matmul(Rg, off[..., c, :].unsqueeze(-1)).squeeze(-1)
                Rg = torch.matmul(Rg, rm[..., c, :, :])
        return jts

    def forward(self, pred_363, gt_363, mask=None):
        B, T, D = pred_363.shape
        losses = {}
        rs, re = G1_363_SLICES['rot_data']
        pred_rot = pred_363[..., rs:re].reshape(B, T, G1_N_JOINTS, 6)
        ric_s, ric_e = G1_363_SLICES['ric_data']
        gt_ric = gt_363[..., ric_s:ric_e].reshape(B, T, G1_N_JOINTS-1, 3)
        pred_ric = pred_363[..., ric_s:ric_e].reshape(B, T, G1_N_JOINTS-1, 3)
        cs, ce = G1_363_SLICES['contacts']
        gt_contacts = gt_363[..., cs:ce]

        fk_pos = self.fk(pred_rot.reshape(B*T, G1_N_JOINTS, 6)).reshape(B, T, G1_N_JOINTS, 3)
        fk_nr = fk_pos[:, :, 1:, :]

        if mask is not None:
            me = mask.unsqueeze(-1).unsqueeze(-1)
            denom = me.sum() * (G1_N_JOINTS-1) + 1e-8
            losses['fk_sup'] = self.sw * ((fk_nr - gt_ric).norm(dim=-1) * me.squeeze(-1)).sum() / denom
            if self.cw > 0:
                losses['fk_sc'] = self.cw * ((fk_nr - pred_ric).norm(dim=-1) * me.squeeze(-1)).sum() / denom
        else:
            losses['fk_sup'] = self.sw * (fk_nr - gt_ric).norm(dim=-1).mean()
            if self.cw > 0:
                losses['fk_sc'] = self.cw * (fk_nr - pred_ric).norm(dim=-1).mean()

        if self.ew > 0:
            ei = self.ee_idx
            fk_ee = fk_pos[:, :, ei, :]
            gt_ee = gt_ric[:, :, [i-1 for i in ei], :]
            if mask is not None:
                ee_d = (fk_ee - gt_ee).norm(dim=-1) * mask.unsqueeze(-1)
                losses['fk_ee'] = self.ew * ee_d.sum() / (mask.sum() * len(ei) + 1e-8)
            else:
                losses['fk_ee'] = self.ew * (fk_ee - gt_ee).norm(dim=-1).mean()

        fk_ee_p = fk_pos[:, :, ei, :]
        fk_ee_v = fk_ee_p[:, 1:] - fk_ee_p[:, :-1]
        fk_ee_s = fk_ee_v.norm(dim=-1)
        ca = gt_contacts[:, 1:] * gt_contacts[:, :-1]
        if mask is not None:
            ca = ca * (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        losses['contact'] = (fk_ee_s * ca).sum() / (ca.sum() + 1e-8)

        bv = []
        for chain in self.tree:
            for j in range(1, len(chain)):
                p, c = chain[j-1], chain[j]
                bl = (fk_pos[:, :, c] - fk_pos[:, :, p]).norm(dim=-1)
                bv.append(bl.var(dim=1).mean())
        if bv:
            losses['bone_len'] = torch.stack(bv).mean() * 0.1

        return losses



def orthogonal_regularization(rot_6d, mask=None):
    '''Orthogonal regularization: directly constrain raw 6D vectors.
    
    For 6D rotation [a1, a2] (both in R^3):
      - ||a1|| should be 1  →  penalty: (||a1||^2 - 1)^2
      - ||a2|| should be 1  →  penalty: (||a2||^2 - 1)^2
      - a1 ⟂ a2             →  penalty: (a1·a2)^2
    
    Args:
        rot_6d: (B, T, J, 6) - raw 6D rotation vectors from decoder
        mask:   (B, T) - valid frame mask
    Returns:
        scalar loss
    '''
    a1 = rot_6d[..., :3]  # (B, T, J, 3)
    a2 = rot_6d[..., 3:]  # (B, T, J, 3)

    # Unit-norm penalties
    norm1 = a1.norm(dim=-1)  # (B, T, J)
    norm2 = a2.norm(dim=-1)  # (B, T, J)
    loss_unit = ((norm1.pow(2) - 1).pow(2) + (norm2.pow(2) - 1).pow(2)).mean(dim=-1)  # (B, T)

    # Orthogonality penalty
    dot = (a1 * a2).sum(dim=-1)  # (B, T, J)
    loss_ortho = (dot.pow(2)).mean(dim=-1)  # (B, T)

    loss_per_frame = loss_unit + loss_ortho  # (B, T)

    if mask is not None:
        return (loss_per_frame * mask).sum() / (mask.sum() + 1e-8)
    return loss_per_frame.mean()



class FKPositionLossQuat(nn.Module):
    """FK-based joint position consistency loss (v5: quaternion rotation).
    
    Uses native quaternions instead of 6D vectors, eliminating the Gram-Schmidt error.
    Losses: fk_sup(supervised) + fk_sc(self-consistent) + fk_ee(end-effector) + contact + bone_len
    """
    def __init__(self, supervised_weight=1.0, self_consist_weight=0.3, ee_weight=2.0):
        super().__init__()
        off = torch.tensor(G1_BONE_OFFSETS, dtype=torch.float32)
        self.register_buffer('offsets', off)
        self.tree = G1_KINEMATIC_TREE
        self.ee_idx = G1_CONTACT_JOINTS
        self.sw, self.cw, self.ew = supervised_weight, self_consist_weight, ee_weight

    @staticmethod
    def quat_to_mat(quat):
        """Convert quaternion (w,x,y,z) to 3x3 rotation matrix.
        Uses the same formula as utils/quaternion.py -> quaternion_to_matrix().
        """
        r, i, j, k = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        two_s = 2.0 / (r*r + i*i + j*j + k*k)
        o00 = 1 - two_s * (j*j + k*k)
        o01 = two_s * (i*j - k*r)
        o02 = two_s * (i*k + j*r)
        o10 = two_s * (i*j + k*r)
        o11 = 1 - two_s * (i*i + k*k)
        o12 = two_s * (j*k - i*r)
        o20 = two_s * (i*k - j*r)
        o21 = two_s * (j*k + i*r)
        o22 = 1 - two_s * (i*i + j*j)
        return torch.stack([
            torch.stack([o00, o01, o02], dim=-1),
            torch.stack([o10, o11, o12], dim=-1),
            torch.stack([o20, o21, o22], dim=-1),
        ], dim=-2)

    def fk(self, quat):
        """Forward kinematics from quaternion rotations.
        
        Args:
            quat: (*, J, 4) --- quaternion rotations per joint (w,x,y,z)
        Returns:
            joint positions (*, J, 3)
        """
        *bd, J, _ = quat.shape
        off = self.offsets.to(quat.device).view(*([1]*len(bd)), J, 3).expand(*bd, -1, -1)
        rm = self.quat_to_mat(quat)
        jts = torch.zeros(*bd, J, 3, device=quat.device)
        for chain in self.tree:
            Rg = rm[..., chain[0], :, :]
            for i in range(1, len(chain)):
                p, c = chain[i-1], chain[i]
                jts[..., c, :] = jts[..., p, :] + torch.matmul(Rg, off[..., c, :].unsqueeze(-1)).squeeze(-1)
                Rg = torch.matmul(Rg, rm[..., c, :, :])
        return jts

    def forward(self, pred_303, gt_303, mask=None):
        """Compute FK losses on 303-dim quaternion data.
        
        Args:
            pred_303: (B, T, 303) --- predicted motion in normalized space
            gt_303:   (B, T, 303) --- ground truth motion in normalized space
            mask:     (B, T) --- valid frame mask
        """
        B, T, D = pred_303.shape
        losses = {}
        qs, qe = G1_303_SLICES['quat_data']
        pred_quat = pred_303[..., qs:qe].reshape(B, T, G1_N_JOINTS, 4)
        ric_s, ric_e = G1_303_SLICES['ric_data']
        gt_ric = gt_303[..., ric_s:ric_e].reshape(B, T, G1_N_JOINTS-1, 3)
        pred_ric = pred_303[..., ric_s:ric_e].reshape(B, T, G1_N_JOINTS-1, 3)
        cs, ce = G1_303_SLICES['contacts']
        gt_contacts = gt_303[..., cs:ce]

        fk_pos = self.fk(pred_quat.reshape(B*T, G1_N_JOINTS, 4)).reshape(B, T, G1_N_JOINTS, 3)
        fk_nr = fk_pos[:, :, 1:, :]

        if mask is not None:
            me = mask.unsqueeze(-1).unsqueeze(-1)
            denom = me.sum() * (G1_N_JOINTS-1) + 1e-8
            losses['fk_sup'] = self.sw * ((fk_nr - gt_ric).norm(dim=-1) * me.squeeze(-1)).sum() / denom
            if self.cw > 0:
                losses['fk_sc'] = self.cw * ((fk_nr - pred_ric).norm(dim=-1) * me.squeeze(-1)).sum() / denom
        else:
            losses['fk_sup'] = self.sw * (fk_nr - gt_ric).norm(dim=-1).mean()
            if self.cw > 0:
                losses['fk_sc'] = self.cw * (fk_nr - pred_ric).norm(dim=-1).mean()

        if self.ew > 0:
            ei = self.ee_idx
            fk_ee = fk_pos[:, :, ei, :]
            gt_ee = gt_ric[:, :, [i-1 for i in ei], :]
            if mask is not None:
                ee_d = (fk_ee - gt_ee).norm(dim=-1) * mask.unsqueeze(-1)
                losses['fk_ee'] = self.ew * ee_d.sum() / (mask.sum() * len(ei) + 1e-8)
            else:
                losses['fk_ee'] = self.ew * (fk_ee - gt_ee).norm(dim=-1).mean()

        # Contact (foot skating) loss
        fk_ee_p = fk_pos[:, :, ei, :]
        fk_ee_v = fk_ee_p[:, 1:] - fk_ee_p[:, :-1]
        fk_ee_s = fk_ee_v.norm(dim=-1)
        ca = gt_contacts[:, 1:] * gt_contacts[:, :-1]
        if mask is not None:
            ca = ca * (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        losses['contact'] = (fk_ee_s * ca).sum() / (ca.sum() + 1e-8)

        # Bone length consistency
        bv = []
        for chain in self.tree:
            for j in range(1, len(chain)):
                p, c = chain[j-1], chain[j]
                bl = (fk_pos[:, :, c] - fk_pos[:, :, p]).norm(dim=-1)
                bv.append(bl.var(dim=1).mean())
        if bv:
            losses['bone_len'] = torch.stack(bv).mean() * 0.1

        return losses
class PhysicsMetrics:
    """Lightweight physics metrics for monitoring (not used in training)."""
    @staticmethod
    def compute(pred, gt, mask=None):
        """Auto-detect 363D vs 303D based on input dimension."""
        D = pred.shape[-1]
        slices = G1_303_SLICES if D == 303 else G1_363_SLICES
        metrics = {}
        ric_s, ric_e = slices['ric_data']
        pred_ric = pred[..., ric_s:ric_e].reshape(*pred.shape[:-1], G1_N_JOINTS-1, 3)
        cs, ce = slices['contacts']
        gt_contacts = gt[..., cs:ce]

        ee_ric_idx = [i-1 for i in G1_CONTACT_JOINTS]
        foot_pos = pred_ric[:, :, ee_ric_idx, :]
        foot_vel = (foot_pos[:, 1:] - foot_pos[:, :-1]).norm(dim=-1)
        ca = gt_contacts[:, 1:] * gt_contacts[:, :-1]
        if mask is not None:
            ca = ca * (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        metrics['foot_skating'] = ((foot_vel * ca).sum() / (ca.sum() + 1e-8)).item()

        vel = pred_ric[:, 1:] - pred_ric[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = (acc[:, 1:] - acc[:, :-1]).norm(dim=-1).mean()
        metrics['jerk'] = jerk.item()

        return metrics