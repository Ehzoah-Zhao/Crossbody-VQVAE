"""convert_6d_to_quat.py - Convert 363D 6D-rotation data to 303D quaternion data

Run on server:
  python convert_6d_to_quat.py

This script:
1. Loads all .npy files from new_joint_vecs_v4/
2. Converts the 6D rotation slice (91:271 = 180D) to quaternion (120D)
3. Saves to new_joint_vecs_v5/
4. Computes new Mean.npy and Std.npy for the 303D format
"""

import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import torch

# ================= CONFIG =================
INPUT_DIR = "./dataset/unitreeg1/new_joint_vecs_v4"   # 363D data (system disk, read-only)
OUTPUT_DIR = "/home/ubuntu/data_drive/dataset/unitreeg1/new_joint_vecs_v5"  # 303D data -> data drive!
META_IN_DIR = "./dataset/unitreeg1/meta_v4"            # old Mean/Std (system disk, read-only)
META_OUT_DIR = "/home/ubuntu/data_drive/dataset/unitreeg1/meta_v5"  # new Mean/Std -> data drive!

N_JOINTS = 30
# 363D slices
ROT_START, ROT_END = 91, 271      # 6D rotations: 30 × 6 = 180D
# New 303D slices
QUAT_START, QUAT_END = 91, 211    # Quaternions: 30 × 4 = 120D
LOCAL_VEL_END_363 = 361
LOCAL_VEL_END_303 = 301
CONTACTS_END_363 = 363
CONTACTS_END_303 = 303
# ===========================================

def rotation_6d_to_matrix(d6):
    """Gram-Schmidt: 6D continuous rotation -> 3x3 rotation matrix."""
    if isinstance(d6, np.ndarray):
        d6 = torch.from_numpy(d6).float()
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def matrix_to_quaternion(matrix):
    """Rotation matrix -> quaternion (w,x,y,z)."""
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    trace = m00 + m11 + m22
    qw = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    scale = 0.25 / qw
    qx = (matrix[..., 2, 1] - matrix[..., 1, 2]) * scale
    qy = (matrix[..., 0, 2] - matrix[..., 2, 0]) * scale
    qz = (matrix[..., 1, 0] - matrix[..., 0, 1]) * scale
    return torch.stack([qw, qx, qy, qz], dim=-1)

def convert_file(filepath, mean_rot, std_rot):
    """Convert a single 363D .npy file to 303D."""
    data = np.load(filepath)  # (T, 363)
    T = data.shape[0]

    # 1. Extract and denormalize 6D rotation
    rot_6d_norm = data[:, ROT_START:ROT_END]  # (T, 180)
    rot_6d_raw = rot_6d_norm * std_rot + mean_rot  # denormalize
    rot_6d = rot_6d_raw.reshape(T, N_JOINTS, 6)

    # 2. 6D -> matrix -> quaternion
    matrix = rotation_6d_to_matrix(rot_6d)  # (T, 30, 3, 3)
    quat = matrix_to_quaternion(matrix)      # (T, 30, 4)
    quat_flat = quat.numpy().reshape(T, -1)  # (T, 120)

    # 3. Reassemble: pre-rot + quat + post-rot
    pre = data[:, :ROT_START]                          # (T, 91)
    post = data[:, ROT_END:]                            # (T, 92) = local_vel(90) + contacts(2)
    new_data = np.concatenate([pre, quat_flat, post], axis=1)  # (T, 303)

    return new_data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(META_OUT_DIR, exist_ok=True)

    # Load old normalization stats
    old_mean = np.load(pjoin(META_IN_DIR, "Mean.npy"))
    old_std = np.load(pjoin(META_IN_DIR, "Std.npy"))
    mean_rot = old_mean[ROT_START:ROT_END]
    std_rot = old_std[ROT_START:ROT_END]

    file_list = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")])
    print(f"Found {len(file_list)} .npy files in {INPUT_DIR}")
    print(f"Input: 363D -> Output: 303D")
    print(f"Rotation: 6D({ROT_END-ROT_START}D) -> Quat({QUAT_END-QUAT_START}D)")

    all_data = []

    for fname in tqdm(file_list, desc="Converting"):
        in_path = pjoin(INPUT_DIR, fname)
        out_path = pjoin(OUTPUT_DIR, fname)

        try:
            new_data = convert_file(in_path, mean_rot, std_rot)
            np.save(out_path, new_data.astype(np.float32))
            all_data.append(new_data)
        except Exception as e:
            print(f"  Error converting {fname}: {e}")
            continue

    if not all_data:
        print("ERROR: No files converted!")
        return

    # Compute new Mean and Std
    all_data = np.concatenate(all_data, axis=0)
    N, D = all_data.shape
    print(f"\nTotal frames: {N}, Dimension: {D}")

    assert D == 303, f"Expected 303D, got {D}D"

    new_mean = all_data.mean(axis=0)
    new_std = all_data.std(axis=0)

    # Fix near-zero std (prevent division by zero)
    tiny_mask = new_std < 1e-4
    if tiny_mask.any():
        print(f"Fixing {tiny_mask.sum()} near-zero std dimensions")
        new_std[tiny_mask] = 1.0

    # Variance balancing (same logic as original cal.py)
    # Group: r_velocity(1) | root_vel_xz(2) | root_y(1) | ric(87) | quat(120) | local_vel(90) | contacts(2)
    groups = [
        (0, 1), (1, 3), (3, 4), (4, 91), (91, 211), (211, 301), (301, 303)
    ]
    for g_start, g_end in groups:
        new_std[g_start:g_end] = new_std[g_start:g_end].mean()

    np.save(pjoin(META_OUT_DIR, "Mean.npy"), new_mean.astype(np.float32))
    np.save(pjoin(META_OUT_DIR, "Std.npy"), new_std.astype(np.float32))

    print(f"\nSaved:")
    print(f"  {OUTPUT_DIR}/  ({len(file_list)} files)")
    print(f"  {META_OUT_DIR}/Mean.npy, Std.npy")
    print(f"\nStd preview:")
    print(f"  r_velocity:  {new_std[0]:.4f}")
    print(f"  root_vel_xz: {new_std[1]:.4f}")
    print(f"  quat(0-3):   {new_std[91:95]}")
    print(f"  contacts:    {new_std[301:303]}")
    print(f"\nDone! Ready for exp_v5_quat training.")

if __name__ == "__main__":
    main()
