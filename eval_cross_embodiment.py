import torch
import numpy as np
import os
from os.path import join as pjoin
import argparse
from tqdm import tqdm

from models.vqvae import CrossEmbodimentVQVAE

# ================= 数学工具 (精确复制自 vis_cross_embodiment.py) =================
def qinv(q):
    a = q.clone(); a[..., 1:] *= -1; return a
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
    return qmul(qmul(q, q_v), qinv(q))[..., 1:]
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
    w = torch.cos(half_yaw); y = torch.sin(half_yaw)
    x = torch.zeros_like(w); z = torch.zeros_like(w)
    return torch.stack([w, x, y, z], dim=-1)

# G1 skeleton: 30 joints, parents derived from edges
G1_SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),
    (15,23),(23,24),(24,25),(25,26),(26,27),(27,28),
]
G1_PARENTS = np.zeros(30, dtype=int)
for (p, c) in G1_SKELETON_EDGES: G1_PARENTS[c] = p

G1_BONE_OFFSETS = torch.tensor([
    [ 0.0,       0.0,       0.0     ],
    [-0.064452, -0.1027,    0.0     ],
    [-0.052,    -0.030465,  0.0     ],
    [ 0.0,      -0.12412,   0.025001],
    [-0.0021489,-0.17734,  -0.078273],
    [ 0.000094, -0.30001,   0.0     ],
    [ 0.0,      -0.017558,  0.0     ],
    [ 0.064452, -0.1027,    0.0     ],
    [ 0.052,    -0.030465,  0.0     ],
    [ 0.0,      -0.12412,   0.025001],
    [ 0.0021489,-0.17734,  -0.078273],
    [-0.000094, -0.30001,   0.0     ],
    [ 0.0,      -0.017558,  0.0     ],
    [ 0.0,       0.0,       0.0     ],
    [ 0.0,       0.044,    -0.003964],
    [ 0.0,       0.0,       0.0     ],
    [-0.10022,   0.24778,   0.003956],
    [-0.038,    -0.013831,  0.0     ],
    [-0.00624,  -0.1032,    0.0     ],
    [ 0.0,      -0.080518,  0.015783],
    [-0.001888, -0.010,     0.100   ],
    [ 0.0,       0.0,       0.038   ],
    [ 0.0,       0.0,       0.046   ],
    [ 0.10021,   0.24778,   0.003956],
    [ 0.038,    -0.013831,  0.0     ],
    [ 0.00624,  -0.1032,    0.0     ],
    [ 0.0,      -0.080518,  0.015783],
    [ 0.001888, -0.010,     0.100   ],
    [ 0.0,       0.0,       0.038   ],
    [ 0.0,       0.0,       0.046   ],
])

def recover_186d_to_positions(features, mean, std):
    """Exact copy from vis_cross_embodiment.py.
    186-dim layout: yaw_vel(1) + root_vel_xz(2) + root_y(1) + rot_6d(180) + contact(2)
    """
    device = features.device
    if mean is not None and std is not None:
        features = features * torch.tensor(std, device=device) + torch.tensor(mean, device=device)

    B, T, D = features.shape

    yaw_vel = features[..., 0:1]
    root_vel_xz = features[..., 1:3]
    root_y = features[..., 3:4]
    rot_data = features[..., 4:184].view(B, T, 30, 6)  # 30 joints x 6D rotation

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
            offset_rotated = qrot(global_rot[:, :, p, :],
                                  true_bone_offsets[i].view(1, 1, 3).repeat(B, T, 1))
            global_pos[:, :, i, :] = global_pos[:, :, p, :] + offset_rotated
    return global_pos


def main():
    parser = argparse.ArgumentParser(description='Cross-Embodiment VQVAE Evaluation')
    parser.add_argument('--ckpt', type=str, default='./output_vqfinal/exp_cross_embodiment/net_iter0160000.pth')
    parser.add_argument('--data_dir_A', type=str, default='./dataset/smpl/new_joint_vecs')
    parser.add_argument('--data_dir_B', type=str, default='./dataset/unitreeg1/new_joint_vecs')
    parser.add_argument('--stat_dir_B', type=str, default='./dataset/unitreeg1/meta')
    parser.add_argument('--split_txt', type=str, default='./dataset/splits/test.txt')
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./output_vqfinal/eval_results')
    args_eval = parser.parse_args()

    device = torch.device(args_eval.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading model from {args_eval.ckpt}...")

    model = CrossEmbodimentVQVAE(
        input_dim_A=134, input_dim_B=186,
        nb_code=512, code_dim=512, output_emb_width=512,
        down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3,
        activation='relu', norm=None, quantizer_type='ema_reset'
    ).to(device)
    ckpt = torch.load(args_eval.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('net', ckpt), strict=True)
    model.eval()
    print("Model loaded successfully.")

    g1_mean = np.load(pjoin(args_eval.stat_dir_B, "Mean.npy"))
    g1_std = np.load(pjoin(args_eval.stat_dir_B, "Std.npy"))

    with open(args_eval.split_txt) as f:
        test_names = [l.strip() for l in f if l.strip()]

    smpl_dir = args_eval.data_dir_A
    g1_dir = args_eval.data_dir_B
    paired_names = [n for n in test_names
                    if os.path.exists(pjoin(smpl_dir, n+'.npy'))
                    and os.path.exists(pjoin(g1_dir, n+'.npy'))]
    unpaired_names = [n for n in test_names
                      if os.path.exists(pjoin(smpl_dir, n+'.npy'))
                      and not os.path.exists(pjoin(g1_dir, n+'.npy'))]

    print(f"Test set: {len(test_names)} total")
    print(f"  Paired (A&B): {len(paired_names)}")
    print(f"  Unpaired (A only): {len(unpaired_names)}")

    # ===== Part 1: Paired Evaluation =====
    print("\n" + "="*60)
    print("PART 1: Paired Data Evaluation (A->B vs GT B)")
    print("="*60)

    paired_samples = paired_names[:args_eval.max_samples]
    all_feat_mse = []
    all_mpjpe = []
    skipped = 0

    for name in tqdm(paired_samples, desc="Evaluating paired"):
        feat_A = np.load(pjoin(smpl_dir, name+'.npy'))
        feat_B_gt = np.load(pjoin(g1_dir, name+'.npy'))

        fA = torch.from_numpy(feat_A).float().unsqueeze(0).to(device)
        with torch.no_grad():
            fB_hat, _, _ = model.forward_AB(fA)
        fB_hat_np = fB_hat.squeeze(0).cpu().numpy()

        # Align temporal lengths
        min_len = min(fB_hat_np.shape[0], feat_B_gt.shape[0])
        if min_len < 10:
            skipped += 1
            continue
        fB_hat_np = fB_hat_np[:min_len]
        feat_B_gt_aligned = feat_B_gt[:min_len]

        # Feature MSE
        feat_mse = np.mean((fB_hat_np - feat_B_gt_aligned) ** 2)
        all_feat_mse.append(feat_mse)

        # 3D MPJPE
        fB_hat_t = torch.from_numpy(fB_hat_np).float().unsqueeze(0)
        fB_gt_t = torch.from_numpy(feat_B_gt_aligned).float().unsqueeze(0)
        pos_pred = recover_186d_to_positions(fB_hat_t, g1_mean, g1_std)
        pos_gt = recover_186d_to_positions(fB_gt_t, g1_mean, g1_std)

        min_len_3d = min(pos_pred.shape[1], pos_gt.shape[1])
        pos_pred = pos_pred[:, :min_len_3d]
        pos_gt = pos_gt[:, :min_len_3d]

        # Root-align for fair comparison
        pos_pred_root = pos_pred[:, :, 0:1, :]
        pos_gt_root = pos_gt[:, :, 0:1, :]
        pos_pred_aligned = pos_pred - pos_pred_root + pos_gt_root

        mpjpe = torch.mean(torch.norm(pos_pred_aligned - pos_gt, dim=-1)).item()
        all_mpjpe.append(mpjpe)

    if skipped:
        print(f"  Skipped {skipped} short samples (< 10 frames)")

    feat_mse_mean = np.mean(all_feat_mse)
    feat_mse_std = np.std(all_feat_mse)
    mpjpe_mean = np.mean(all_mpjpe)
    mpjpe_std = np.std(all_mpjpe)

    print(f"\nResults on {len(all_mpjpe)} paired samples:")
    print(f"  Feature MSE:      {feat_mse_mean:.6f} +- {feat_mse_std:.6f}")
    print(f"  3D MPJPE (cm):    {mpjpe_mean*100:.2f} +- {mpjpe_std*100:.2f}")

    # ===== Part 2: Unpaired Generation =====
    print("\n" + "="*60)
    print("PART 2: Unpaired A-Only Generation")
    print("="*60)

    os.makedirs(args_eval.save_dir, exist_ok=True)
    unpaired_samples = unpaired_names[:min(50, len(unpaired_names))]

    saved = 0
    for name in tqdm(unpaired_samples, desc="Generating unpaired"):
        feat_A = np.load(pjoin(smpl_dir, name+'.npy'))
        fA = torch.from_numpy(feat_A).float().unsqueeze(0).to(device)
        with torch.no_grad():
            fB_hat, _, _ = model.forward_AB(fA)
        pos_pred = recover_186d_to_positions(fB_hat.cpu(), g1_mean, g1_std)
        np.save(pjoin(args_eval.save_dir, f"gen_{name}.npy"), pos_pred.squeeze(0).numpy())
        saved += 1

    print(f"Saved {saved} generated G1 motions to {args_eval.save_dir}/")

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"  Paired Feature MSE:   {feat_mse_mean:.6f}")
    print(f"  Paired 3D MPJPE:      {mpjpe_mean*100:.2f} cm")
    print(f"  Generated motions:     {args_eval.save_dir}/gen_*.npy")

if __name__ == "__main__":
    main()
