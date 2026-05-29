"""feature_to_joints_v5_quat.py - Recover global 3D joints from 303-dim quaternion features

303-dim layout (v5): r_velocity(1)+root_vel_xz(2)+root_y(1)+ric(87)+quat(120)+local_vel(90)+contacts(2)

Key difference from v4: rotations are already quaternions, no 6D->matrix->quat conversion needed.
"""

import torch
import numpy as np

# ================= Core Math =================
def qinv(q):
    a = q.clone()
    a[..., 1:] *= -1
    return a

def qmul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def qrot(q, v):
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_inv_val = qinv(q)
    res = qmul(qmul(q, q_v), q_inv_val)
    return res[..., 1:]

def yaw_to_quaternion(yaw):
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    y = torch.sin(half_yaw)
    x = torch.zeros_like(w)
    z = torch.zeros_like(w)
    return torch.stack([w, x, y, z], dim=-1)

def quat_normalize(q):
    """Normalize quaternion to unit length."""
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

# ================= Core Recovery Logic v5 (Quat) =================
def recover_g1_motion_v5(features, mean=None, std=None):
    """
    Recover global 3D positions and global quaternion rotations from v5 (303-dim) features.
    
    303-dim layout:
      r_velocity(1) + root_vel_xz(2) + root_y(1) + ric(87=29x3) + quat(120=30x4) + local_vel(90) + contacts(2)
    
    Args:
        features: (B, T, 303) normalized feature tensor
        mean: (303,) numpy array for denormalization
        std:  (303,) numpy array for denormalization
    Returns:
        global_positions: (B, T, 30, 3)
        global_rotations: (B, T, 30, 4) unit quaternions (w,x,y,z)
    """
    if mean is not None and std is not None:
        device = features.device
        t_mean = torch.tensor(mean, device=device, dtype=features.dtype)
        t_std = torch.tensor(std, device=device, dtype=features.dtype)
        features = features * t_std + t_mean

    B, T, D = features.shape
    
    # Auto-detect joint count from 303D format
    # D = 3 (root_vel+yaw+y) + (J-1)*3 (ric) + J*4 (quat) + (J-1)*3 (local_vel) + 2 (contacts)
    # = 3 + (J-1)*3 + J*4 + (J-1)*3 + 2 = 3 + 3J -3 + 4J + 3J - 3 + 2 = 10J - 1
    # For J=30: 10*30 - 1 = 299... hmm wait let me recalculate.
    # Actually: r_velocity(1) + root_vel_xz(2) + root_y(1) + ric(29*3=87) + quat(30*4=120) + local_vel(30*3=90) + contacts(2)
    # = 1 + 2 + 1 + 87 + 120 + 90 + 2 = 303. Correct.
    # So J = 30. Hardcode for G1.
    J = 30

    # Dimension slices (denormalized space)
    idx_yaw_vel_end = 1           # 0:1
    idx_root_vel_end = 3          # 1:3
    idx_root_y_end = 4            # 3:4
    idx_ric_end = 4 + (J-1) * 3   # 4:91
    idx_quat_end = idx_ric_end + J * 4  # 91:211
    
    yaw_vel = features[..., 0:1]
    root_vel_xz = features[..., 1:3]
    root_y = features[..., 3:4]
    ric_data = features[..., 4:idx_ric_end].reshape(B, T, J-1, 3)
    # quaternions are in normalized space - denormalized quat may not be unit
    quat_data = features[..., idx_ric_end:idx_quat_end].reshape(B, T, J, 4)

    root_rots_yaw_only = torch.zeros((B, T, 4), device=features.device)
    root_pos = torch.zeros((B, T, 3), device=features.device)

    q_curr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=features.device).repeat(B, 1)
    pos_curr = torch.zeros((B, 3), device=features.device)

    # 1. Frame-by-frame root trajectory (Yaw-only base)
    for t in range(T):
        if t > 0:
            q_delta = yaw_to_quaternion(yaw_vel[:, t-1, 0])
            q_curr = qmul(q_curr, q_delta)
            q_curr = quat_normalize(q_curr)
            
            v_loc_xz = root_vel_xz[:, t-1]
            v_loc_3d = torch.stack([v_loc_xz[:, 0], torch.zeros_like(v_loc_xz[:, 0]), v_loc_xz[:, 1]], dim=-1)
            q_prev = root_rots_yaw_only[:, t-1]
            v_glob = qrot(q_prev, v_loc_3d)
            
            pos_curr[:, 0] += v_glob[:, 0]
            pos_curr[:, 2] += v_glob[:, 2]
        
        pos_curr[:, 1] = root_y[:, t, 0]
        root_rots_yaw_only[:, t] = q_curr.clone()
        root_pos[:, t] = pos_curr.clone()

    # 2. Recover global rotations: fuse pelvis Yaw with relative quaternions
    # quat_data contains 30 joints of relative rotations in quaternion form
    # The root joint quaternion needs to be combined with pelvis yaw
    joint_rot_relative = quat_normalize(quat_data)
    
    q_root_expanded_for_rot = root_rots_yaw_only.unsqueeze(2).repeat(1, 1, J, 1)
    # For the root joint (index 0), the relative rotation IS the yaw rotation
    # For other joints, compose: global = root_yaw * relative
    global_rotations = qmul(q_root_expanded_for_rot, joint_rot_relative)
    global_rotations = quat_normalize(global_rotations)
    
    # 3. Recover global positions
    global_positions = torch.zeros((B, T, J, 3), device=features.device)
    global_positions[:, :, 0, :] = root_pos
    
    q_root_expanded_for_pos = root_rots_yaw_only.unsqueeze(2).repeat(1, 1, J-1, 1)
    local_pos = qrot(q_root_expanded_for_pos, ric_data)
    global_positions[:, :, 1:, :] = root_pos.unsqueeze(2) + local_pos
    
    return global_positions, global_rotations

# ================= Test =================
if __name__ == "__main__":
    B, T, D = 2, 10, 303  # G1 30 joints -> 303D
    dummy_features = torch.randn(B, T, D)
    
    positions, rotations = recover_g1_motion_v5(dummy_features)
    print("Successfully recovered v5 features (quaternion version)!")
    print(f"Global positions shape: {positions.shape}  (B, T, J, 3)")
    print(f"Global rotations shape: {rotations.shape}  (B, T, J, 4)")