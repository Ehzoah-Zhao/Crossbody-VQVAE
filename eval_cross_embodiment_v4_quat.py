"""
eval_cross_embodiment_v4_quat.py - Evaluation for 303-dim quaternion cross-embodiment VQ-VAE

Metrics: Feature MSE, 3D MPJPE (via FK with quaternions), Foot Skating, Jerk
Supports paired and unpaired evaluation.
"""

import torch, numpy as np, os, argparse
from os.path import join as pjoin
from tqdm import tqdm
import torch.nn.functional as F

from models.vqvae import CrossEmbodimentVQVAE
from utils.physics_losses import FKPositionLossQuat, PhysicsMetrics, G1_303_SLICES, G1_N_JOINTS, G1_CONTACT_JOINTS


def load_model(ckpt_path, device):
    model = CrossEmbodimentVQVAE(
        input_dim_A=134, input_dim_B=303,
        nb_code=512, code_dim=512, output_emb_width=512,
        down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3,
        activation="relu", norm=None, quantizer_type="ema_reset",
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("net", ckpt), strict=True)
    model.eval()
    return model


def evaluate_paired(model, smpl_dir, g1_dir, g1_mean, g1_std, test_names, device, fk_fn, max_samples=100):
    """A->B translation vs GT B."""
    paired = [n for n in test_names if os.path.exists(pjoin(smpl_dir, n+".npy"))
              and os.path.exists(pjoin(g1_dir, n+".npy"))]
    samples = paired[:max_samples]

    feat_mses, mpjpes, foot_skatings, jerks = [], [], [], []
    skipped = 0

    for name in tqdm(samples, desc="Paired eval"):
        fA = np.load(pjoin(smpl_dir, name+".npy"))
        fB_gt = np.load(pjoin(g1_dir, name+".npy"))

        fA_t = torch.from_numpy(fA).float().unsqueeze(0).to(device)
        fB_gt_t = torch.from_numpy(fB_gt).float().unsqueeze(0).to(device)

        with torch.no_grad():
            fB_hat, _, _ = model.forward_AB(fA_t)

        min_len = min(fB_hat.shape[1], fB_gt_t.shape[1])
        if min_len < 10:
            skipped += 1; continue

        fB_hat_aligned = fB_hat[:, :min_len]
        fB_gt_aligned = fB_gt_t[:, :min_len]

        # Feature MSE
        feat_mses.append(F.mse_loss(fB_hat_aligned, fB_gt_aligned).item())

        # 3D MPJPE via FK
        rs, re = G1_303_SLICES["quat_data"]
        pred_quat = fB_hat_aligned[..., rs:re].reshape(1, min_len, G1_N_JOINTS, 4)
        gt_quat = fB_gt_aligned[..., rs:re].reshape(1, min_len, G1_N_JOINTS, 4)

        fk_pred = fk_fn.fk(pred_quat.reshape(min_len, G1_N_JOINTS, 4)).reshape(1, min_len, G1_N_JOINTS, 3)
        fk_gt = fk_fn.fk(gt_quat.reshape(min_len, G1_N_JOINTS, 4)).reshape(1, min_len, G1_N_JOINTS, 3)

        mpjpe = (fk_pred - fk_gt).norm(dim=-1).mean().item()
        mpjpes.append(mpjpe)

        # Physics metrics
        phy = PhysicsMetrics.compute(fB_hat_aligned, fB_gt_aligned)
        foot_skatings.append(phy.get("foot_skating", 0))
        jerks.append(phy.get("jerk", 0))

    print("  Skipped %d short samples" % skipped)
    print("  Feature MSE:    %.6f" % np.mean(feat_mses))
    print("  3D MPJPE (m):   %.4f" % np.mean(mpjpes))
    print("  Foot Skating:   %.6f" % np.mean(foot_skatings))
    print("  Jerk:           %.6f" % np.mean(jerks))

    return dict(feat_mse=np.mean(feat_mses), mpjpe=np.mean(mpjpes),
                foot_skating=np.mean(foot_skatings), jerk=np.mean(jerks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir_A", type=str, required=True)
    parser.add_argument("--data_dir_B", type=str, required=True)
    parser.add_argument("--stat_dir_B", type=str, required=True)
    parser.add_argument("--split_txt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./eval_results_v3")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args_eval = parser.parse_args()

    device = torch.device(args_eval.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(args_eval.ckpt, device)
    fk_fn = FKPositionLossQuat().to(device)

    with open(args_eval.split_txt) as f:
        test_names = [l.strip() for l in f if l.strip()]

    g1_mean = np.load(pjoin(args_eval.stat_dir_B, "Mean.npy"))
    g1_std = np.load(pjoin(args_eval.stat_dir_B, "Std.npy"))

    # Paired evaluation
    print("=" * 60)
    print("Paired Evaluation (A->B vs GT B)")
    print("=" * 60)
    results = evaluate_paired(model, args_eval.data_dir_A, args_eval.data_dir_B,
                               g1_mean, g1_std, test_names, device, fk_fn, args_eval.max_samples)

    # Unpaired generation
    print("\n" + "=" * 60)
    print("Unpaired Generation (A only -> B)")
    print("=" * 60)
    os.makedirs(args_eval.save_dir, exist_ok=True)
    unpaired = [n for n in test_names
                if os.path.exists(pjoin(args_eval.data_dir_A, n+".npy"))
                and not os.path.exists(pjoin(args_eval.data_dir_B, n+".npy"))]
    saved = 0
    for name in tqdm(unpaired[:50], desc="Generating"):
        fA = np.load(pjoin(args_eval.data_dir_A, name+".npy"))
        fA_t = torch.from_numpy(fA).float().unsqueeze(0).to(device)
        with torch.no_grad():
            fB_hat, _, _ = model.forward_AB(fA_t)
        np.save(pjoin(args_eval.save_dir, "gen_%s.npy" % name), fB_hat.squeeze(0).cpu().numpy())
        saved += 1
    print("Generated %d motions -> %s/" % (saved, args_eval.save_dir))


if __name__ == "__main__":
    main()