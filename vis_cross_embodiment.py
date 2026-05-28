import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
from os.path import join as pjoin

from models.vqvae import CrossEmbodimentVQVAE
import options.option_vq as option_vq

# ================= 核心配置区 =================
TARGET_NAME = '000002'         # 默认测试动作；命令行: python vis_cross_embodiment.py 000421
SMPL_FEAT_DIR = 'dataset/smpl/new_joint_vecs'
SMPL_JOINTS_DIR = 'dataset/humanml3d/new_joints'

CKPT_PATH = './output_vqfinal/exp_cross_embodiment/net_iter0190000.pth'
MEAN_STD_PATH_G1 = 'dataset/unitreeg1/meta'
# ===============================================

# ================= 骨架拓扑与 URDF 真实偏移 =================
SMPL_SKELETON_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21)
]

G1_SKELETON_EDGES = [
    (0,1), (1,2), (2,3), (3,4), (4,5), (5,6),
    (0,7), (7,8), (8,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15),
    (15,16), (16,17), (17,18), (18,19), (19,20), (20,21), (21,22),
    (15,23), (23,24), (24,25), (25,26), (26,27), (27,28),
]

G1_PARENTS = np.zeros(30, dtype=int)
for (parent, child) in G1_SKELETON_EDGES:
    G1_PARENTS[child] = parent

# 从 URDF 坐标系 (X=forward, Y=left, Z=up) 转换为 motion 坐标系 (Y=up)
# motion[x, y, z] = [-urdf_y, urdf_z, urdf_x]
G1_BONE_OFFSETS = torch.tensor([
    [ 0.0,       0.0,       0.0     ],  # 0:  Pelvis
    [-0.064452, -0.1027,    0.0     ],  # 1:  L Hip Pitch
    [-0.052,    -0.030465,  0.0     ],  # 2:  L Hip Roll
    [ 0.0,      -0.12412,   0.025001],  # 3:  L Hip Yaw
    [-0.0021489,-0.17734,  -0.078273],  # 4:  L Knee
    [ 0.000094, -0.30001,   0.0     ],  # 5:  L Ankle Pitch
    [ 0.0,      -0.017558,  0.0     ],  # 6:  L Ankle Roll
    [ 0.064452, -0.1027,    0.0     ],  # 7:  R Hip Pitch
    [ 0.052,    -0.030465,  0.0     ],  # 8:  R Hip Roll
    [ 0.0,      -0.12412,   0.025001],  # 9:  R Hip Yaw
    [ 0.0021489,-0.17734,  -0.078273],  # 10: R Knee
    [-0.000094, -0.30001,   0.0     ],  # 11: R Ankle Pitch
    [ 0.0,      -0.017558,  0.0     ],  # 12: R Ankle Roll
    [ 0.0,       0.0,       0.0     ],  # 13: Waist Yaw
    [ 0.0,       0.044,    -0.003964],  # 14: Waist Roll
    [ 0.0,       0.0,       0.0     ],  # 15: Waist Pitch (Torso)
    [-0.10022,   0.24778,   0.003956],  # 16: L Shoulder Pitch
    [-0.038,    -0.013831,  0.0     ],  # 17: L Shoulder Roll
    [-0.00624,  -0.1032,    0.0     ],  # 18: L Shoulder Yaw
    [ 0.0,      -0.080518,  0.015783],  # 19: L Elbow
    [-0.001888, -0.010,     0.100   ],  # 20: L Wrist Roll
    [ 0.0,       0.0,       0.038   ],  # 21: L Wrist Pitch
    [ 0.0,       0.0,       0.046   ],  # 22: L Wrist Yaw
    [ 0.10021,   0.24778,   0.003956],  # 23: R Shoulder Pitch
    [ 0.038,    -0.013831,  0.0     ],  # 24: R Shoulder Roll
    [ 0.00624,  -0.1032,    0.0     ],  # 25: R Shoulder Yaw
    [ 0.0,      -0.080518,  0.015783],  # 26: R Elbow
    [ 0.001888, -0.010,     0.100   ],  # 27: R Wrist Roll
    [ 0.0,       0.0,       0.038   ],  # 28: R Wrist Pitch
    [ 0.0,       0.0,       0.046   ],  # 29: R Wrist Yaw
])

# ================= 核心数学工具 =================
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
    res = qmul(qmul(q, q_v), qinv(q))
    return res[..., 1:]

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def matrix_to_quaternion(matrix):
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    scale = 0.25 / torch.clamp(w, min=1e-8)
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) * scale
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) * scale
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) * scale
    return torch.stack([w, x, y, z], dim=-1)

def yaw_to_quaternion(yaw):
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    y = torch.sin(half_yaw)
    x = torch.zeros_like(w)
    z = torch.zeros_like(w)
    return torch.stack([w, x, y, z], dim=-1)

# ================= FK: 186D 特征 -> 30 关节 3D 位置 =================
def recover_186d_to_positions(features, mean, std):
    device = features.device
    if mean is not None and std is not None:
        features = features * torch.tensor(std, device=device) + torch.tensor(mean, device=device)

    B, T, D = features.shape

    yaw_vel = features[..., 0:1]
    root_vel_xz = features[..., 1:3]
    root_y = features[..., 3:4]
    rot_data = features[..., 4:184].view(B, T, 30, 6)

    root_rots_yaw_only = torch.zeros((B, T, 4), device=device)
    root_pos = torch.zeros((B, T, 3), device=device)
    q_curr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(B, 1)
    pos_curr = torch.zeros((B, 3), device=device)

    for t in range(T):
        if t > 0:
            q_delta = yaw_to_quaternion(yaw_vel[:, t-1, 0])
            q_curr = qmul(q_curr, q_delta)
            q_curr = q_curr / q_curr.norm(dim=-1, keepdim=True)

            v_loc_xz = root_vel_xz[:, t-1]
            v_loc_3d = torch.stack([v_loc_xz[:, 0], torch.zeros_like(v_loc_xz[:, 0]), v_loc_xz[:, 1]], dim=-1)
            v_glob = qrot(root_rots_yaw_only[:, t-1], v_loc_3d)

            pos_curr[:, 0] += v_glob[:, 0]
            pos_curr[:, 2] += v_glob[:, 2]

        pos_curr[:, 1] = root_y[:, t, 0]
        root_rots_yaw_only[:, t] = q_curr.clone()
        root_pos[:, t] = pos_curr.clone()

    local_rot_mat = rotation_6d_to_matrix(rot_data)
    local_quats = matrix_to_quaternion(local_rot_mat)

    global_pos = torch.zeros((B, T, 30, 3), device=device)
    global_rot = torch.zeros((B, T, 30, 4), device=device)

    true_bone_offsets = G1_BONE_OFFSETS.to(device)

    for i in range(30):
        if i == 0:
            global_pos[:, :, 0, :] = root_pos
            global_rot[:, :, 0, :] = qmul(root_rots_yaw_only, local_quats[:, :, 0, :])
        else:
            p = G1_PARENTS[i]
            global_rot[:, :, i, :] = qmul(global_rot[:, :, p, :], local_quats[:, :, i, :])
            offset_rotated = qrot(global_rot[:, :, p, :], true_bone_offsets[i].view(1, 1, 3).repeat(B, T, 1))
            global_pos[:, :, i, :] = global_pos[:, :, p, :] + offset_rotated

    return global_pos

# ================= 绘图主程序 =================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 支持命令行指定目标文件
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET_NAME
    print(f'[Cross-Embodiment Viz] Target: {target}  |  Device: {device}')

    smpl_joints_gt = np.load(pjoin(SMPL_JOINTS_DIR, target + '.npy'))
    smpl_feat = np.load(pjoin(SMPL_FEAT_DIR, target + '.npy'))
    feat_A = torch.from_numpy(smpl_feat).float().unsqueeze(0).to(device)

    # 安全解析参数
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    args = option_vq.get_args_parser()
    sys.argv = saved_argv

    model = CrossEmbodimentVQVAE(
        input_dim_A=134,
        input_dim_B=186,
        nb_code=getattr(args, 'nb_code', 512),
        code_dim=getattr(args, 'code_dim', 512),
        output_emb_width=getattr(args, 'output_emb_width', 512),
        down_t=getattr(args, 'down_t', 3),
        stride_t=getattr(args, 'stride_t', 2),
        width=getattr(args, 'width', 512),
        depth=getattr(args, 'depth', 3),
        dilation_growth_rate=getattr(args, 'dilation_growth_rate', 3),
        activation=getattr(args, 'activation', 'relu'),
        norm=getattr(args, 'norm', None),
        quantizer_type='ema_reset'
    ).to(device)

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint.get('net', checkpoint), strict=True)
    model.eval()

    with torch.no_grad():
        feat_B_hat, _, _ = model.forward_AB(feat_A)

    t_mean_g1 = np.load(pjoin(MEAN_STD_PATH_G1, 'Mean.npy'))
    t_std_g1 = np.load(pjoin(MEAN_STD_PATH_G1, 'Std.npy'))

    g1_joints_pred = recover_186d_to_positions(feat_B_hat, t_mean_g1, t_std_g1)[0].cpu().numpy()

    T = min(smpl_joints_gt.shape[0], g1_joints_pred.shape[0])

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.suptitle(f'Cross-Embodiment: SMPL -> G1  |  {target}', fontsize=14)

    def init_axes(ax, title, current_root):
        radius = 1.0
        ax.set_xlim3d([current_root[0] - radius, current_root[0] + radius])
        ax.set_ylim3d([current_root[2] - radius, current_root[2] + radius])
        ax.set_zlim3d([current_root[1] - 0.2, current_root[1] + 1.5])
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=20, azim=-60)
        ax.axis('off')

    def draw_skeleton(ax, pos, edges, is_g1=False):
        for i, (j1, j2) in enumerate(edges):
            if j1 < pos.shape[0] and j2 < pos.shape[0]:
                if is_g1:
                    if i < 6:       c = '#E53935'
                    elif i < 12:    c = '#1E88E5'
                    elif i < 15:    c = '#424242'
                    elif i < 22:    c = '#FB8C00'
                    else:           c = '#43A047'
                else:
                    if i < 4:       c = '#E53935'
                    elif i < 8:     c = '#1E88E5'
                    elif i < 12:    c = '#424242'
                    elif i < 16:    c = '#FB8C00'
                    else:           c = '#43A047'
                ax.plot([pos[j1,0], pos[j2,0]],
                        [pos[j1,2], pos[j2,2]],
                        [pos[j1,1], pos[j2,1]], c=c, lw=2)
        ax.scatter(pos[0,0], pos[0,2], pos[0,1], c='red', marker='s', s=40)

    def update(frame):
        ax1.cla()
        init_axes(ax1, f'SMPL GT (Frame {frame})', smpl_joints_gt[frame, 0])
        draw_skeleton(ax1, smpl_joints_gt[frame], SMPL_SKELETON_EDGES, is_g1=False)

        ax2.cla()
        init_axes(ax2, f'G1 Generated (Frame {frame})', g1_joints_pred[frame, 0])
        draw_skeleton(ax2, g1_joints_pred[frame], G1_SKELETON_EDGES, is_g1=True)

    ani = FuncAnimation(fig, update, frames=T, interval=50)
    plt.show()

if __name__ == '__main__':
    main()
