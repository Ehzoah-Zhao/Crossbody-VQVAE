""" diagnose_coord.py - quick coordinate system / FK diagnostic for a single paired sample
Usage on server:
  python diagnose_coord.py --ckpt output_vqfinal/exp_v3_363fk_cut/net_iter0150000.pth
"""
import torch, numpy as np, argparse, os
from os.path import join as pjoin

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--sample_name", type=str, default="000021")
parser.add_argument("--data_dir_A", type=str, default="./dataset/smpl/new_joint_vecs")
parser.add_argument("--data_dir_B", type=str, default="./dataset/unitreeg1/new_joint_vecs_v4")
args = parser.parse_args()

from models.vqvae import CrossEmbodimentVQVAE
from utils.physics_losses import FKPositionLoss, G1_363_SLICES, G1_N_JOINTS, G1_BONE_OFFSETS

device = torch.device("cuda")
model = CrossEmbodimentVQVAE(
    input_dim_A=134, input_dim_B=363, nb_code=512, code_dim=512, output_emb_width=512,
    down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
    activation="relu", norm=None, quantizer_type="ema_reset",
).to(device)
ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt.get("net", ckpt), strict=True)
model.eval()

fk_fn = FKPositionLoss().to(device)
rs, re = G1_363_SLICES["rot_data"]

name = args.sample_name
fA = np.load(pjoin(args.data_dir_A, name+".npy"))
fB_gt = np.load(pjoin(args.data_dir_B, name+".npy"))

fA_t = torch.from_numpy(fA).float().unsqueeze(0).to(device)
fB_gt_t = torch.from_numpy(fB_gt).float().unsqueeze(0).to(device)

with torch.no_grad():
    fB_hat, _, _ = model.forward_AB(fA_t)

T = min(fB_hat.shape[1], fB_gt_t.shape[1])
pred = fB_hat[:, :T]
gt = fB_gt_t[:, :T]

# FK on predicted rotations
pred_rot = pred[..., rs:re].reshape(1, T, G1_N_JOINTS, 6)
gt_rot = gt[..., rs:re].reshape(1, T, G1_N_JOINTS, 6)

fk_pred = fk_fn.fk(pred_rot.reshape(T, G1_N_JOINTS, 6)).reshape(1, T, G1_N_JOINTS, 3)
fk_gt = fk_fn.fk(gt_rot.reshape(T, G1_N_JOINTS, 6)).reshape(1, T, G1_N_JOINTS, 3)

print("=" * 60)
print("FK Coordinate System Diagnostic")
print("=" * 60)

# 1. Check rotation norm
pred_rot_norm = pred_rot.norm(dim=-1).mean()
gt_rot_norm = gt_rot.norm(dim=-1).mean()
print(f"\n1. Rotation norm (mean over joints/time):")
print(f"   Pred: {pred_rot_norm:.4f}  |  GT: {gt_rot_norm:.4f}")

# 2. Check FK positions range
print(f"\n2. FK position ranges (over first frame):")
for label, fk in [("Pred", fk_pred[0,0]), ("GT", fk_gt[0,0])]:
    print(f"   {label}: X=[{fk[:,0].min():.3f}, {fk[:,0].max():.3f}], "
          f"Y=[{fk[:,1].min():.3f}, {fk[:,1].max():.3f}], "
          f"Z=[{fk[:,2].min():.3f}, {fk[:,2].max():.3f}]")

# 3. Root joint position
print(f"\n3. Root joint (idx 0) FK position (first frame):")
print(f"   Pred: {fk_pred[0,0,0].cpu().numpy()}")
print(f"   GT:   {fk_gt[0,0,0].cpu().numpy()}")

# 4. Check if pred is just rotated relative to GT (Procrustes)
from scipy.linalg import orthogonal_procrustes
pred_flat = fk_pred[0].reshape(-1, 3).cpu().numpy()
gt_flat = fk_gt[0].reshape(-1, 3).cpu().numpy()
R, scale = orthogonal_procrustes(pred_flat, gt_flat)
aligned = pred_flat @ R * scale
mpjpe_raw = np.linalg.norm(pred_flat - gt_flat, axis=-1).mean()
mpjpe_aligned = np.linalg.norm(aligned - gt_flat, axis=-1).mean()
print(f"\n4. Procrustes analysis:")
print(f"   Raw MPJPE (no alignment): {mpjpe_raw:.4f} m")
print(f"   MPJPE after Procrustes:   {mpjpe_aligned:.4f} m")
print(f"   Rotation matrix R:\n{R}")

# 5. Key conclusion
if mpjpe_aligned < 0.10:
    print(f"\n>>> CONCLUSION: After Procrustes alignment, MPJPE drops from {mpjpe_raw:.2f}m to {mpjpe_aligned:.2f}m.")
    print(f">>> This means the model IS learning correct joint structure, but in a ROTATED coordinate frame!")
    print(f">>> Fix: need to align coordinate systems between SMPL and G1 data preprocessing.")
else:
    print(f"\n>>> CONCLUSION: Even after Procrustes alignment, MPJPE = {mpjpe_aligned:.2f}m.")
    print(f">>> Problem is NOT just rotation - model genuinely can't reconstruct joint positions.")
    print(f">>> Likely root cause: 6D rotation non-orthogonality causing FK error accumulation.")
