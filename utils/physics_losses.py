"""utils/physics_losses.py - FK-based physical consistency losses for 363-dim G1 data

363-dim layout: r_velocity(1)+root_vel_xz(2)+root_y(1)+ric(87)+rot(180)+local_vel(90)+contacts(2)
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


class PhysicsMetrics:
    """Lightweight physics metrics for monitoring (not used in training)."""
    @staticmethod
    def compute(pred_363, gt_363, mask=None):
        metrics = {}
        ric_s, ric_e = G1_363_SLICES['ric_data']
        pred_ric = pred_363[..., ric_s:ric_e].reshape(*pred_363.shape[:-1], G1_N_JOINTS-1, 3)
        cs, ce = G1_363_SLICES['contacts']
        gt_contacts = gt_363[..., cs:ce]

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