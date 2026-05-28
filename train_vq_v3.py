"""
train_vq_v3.py - Cross-Embodiment VQ-VAE with P0+P1 improvements

P0: 363-dim G1 + FK physical losses + CUT unpaired contrastive
P1: Physics metrics monitoring

Stage 1 (warm-up): reconstruction only
Stage 2 (main): + cycle + CUT contrastive
Stage 3 (main): + paired contrastive + FK losses

Usage:
  python train_vq_v3.py --data_dir_A ./dataset/smpl/new_joint_vecs 
      --data_dir_B ./dataset/unitreeg1/new_joint_vecs_v4 
      --stat_dir_A ./dataset/smpl/meta --stat_dir_B ./dataset/unitreeg1/meta_v4 
      --split_txt_dir ./dataset/splits --exp-name exp_v3
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.utils_model as utils_model
import options.option_vq as option_vq
from utils.cross_losses import MotionReconLoss, ContrastiveLoss
from utils.physics_losses import FKPositionLoss, PhysicsMetrics
from utils.unpaired_contrastive import UnpairedContrastiveLoss

from dataset.dataset_A import DATALoader as DataLoaderA
from dataset.dataset_B_v4 import DATALoader as DataLoaderB
from dataset.paired_dataset import PairedDATALoader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for pg in optimizer.param_groups:
        pg["lr"] = current_lr
    return optimizer, current_lr


def make_zero_stats(*keys):
    return {k: 0. for k in keys}


def get_weights(nb_iter, warm_up_iter):
    """Stage-based curriculum learning."""
    if nb_iter < warm_up_iter:
        return dict(recon=1.0, vel=0.5, commit=0.02, cycle=0.0, contrast=0.0,
                    unp_contrast=0.0, fk_sup=0.0, fk_sc=0.0, fk_ee=0.0,
                    contact=0.0, bone_len=0.0)
    elif nb_iter < warm_up_iter * 2:
        return dict(recon=1.0, vel=0.5, commit=0.02, cycle=0.1, contrast=0.0,
                    unp_contrast=0.05, fk_sup=0.0, fk_sc=0.0, fk_ee=0.0,
                    contact=0.0, bone_len=0.0)
    else:
        return dict(recon=1.0, vel=0.5, commit=0.02, cycle=0.1, contrast=0.05,
                    unp_contrast=0.05, fk_sup=0.3, fk_sc=0.1, fk_ee=0.5,
                    contact=0.1, bone_len=0.05)


if __name__ == "__main__":
    # ========== 1. Config ==========
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    import wandb
    wandb.init(project="Crossbody-VQVAE-v3", name=args.exp_name, config=vars(args))

    # ========== 2. Datasets ==========
    train_txt = os.path.join(args.split_txt_dir, "train.txt")
    logger.info("Loading Datasets...")

    # A: SMPL (134-dim)
    train_loader_A = DataLoaderA(args.data_dir_A, train_txt, args.stat_dir_A, args.batch_size, args.window_size)
    # B: G1 v4 (363-dim)
    train_loader_B = DataLoaderB(args.data_dir_B, train_txt, args.stat_dir_B, args.batch_size, args.window_size)
    # Paired: same data dirs (paired by filename matching)
    paired_loader = PairedDATALoader(args.data_dir_A, args.data_dir_B,
                                     train_txt, args.stat_dir_A, args.stat_dir_B,
                                     args.batch_size, args.window_size)

    iter_A = cycle(train_loader_A)
    iter_B = cycle(train_loader_B)
    iter_paired = cycle(paired_loader)

    logger.info("A: %d  B: %d  Paired: %d", len(train_loader_A.dataset),
                len(train_loader_B.dataset), len(paired_loader.dataset))

    # ========== 3. Model (363-dim B) ==========
    net = vqvae.CrossEmbodimentVQVAE(
        input_dim_A=134, input_dim_B=363,
        nb_code=args.nb_code, code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t, stride_t=args.stride_t,
        width=args.width, depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=getattr(args, "activation", getattr(args, "vq_act", "relu")), norm=getattr(args, "norm", getattr(args, "vq_norm", None)),
        quantizer_type=args.quantizer_type, mu=args.mu,
    ).cuda()

    # ========== 4. Losses ==========
    recon_loss = MotionReconLoss(loss_type="l1_smooth")
    contrastive_loss = ContrastiveLoss(temperature=0.07)
    unp_contrast_loss = UnpairedContrastiveLoss(temperature=0.07, num_patches=8)
    fk_loss_fn = FKPositionLoss(supervised_weight=1.0, self_consist_weight=0.3, ee_weight=2.0)

    # ========== 5. Optimizer ==========
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter)

    # ========== 6. Training loop ==========
    STAT_KEYS = ["recon_A","recon_B","vel_A","vel_B","cyc_ABA","cyc_BAB",
                 "contrast","unp_contrast","fk_sup","fk_sc","fk_ee",
                 "contact","bone_len","commit","ppl_A","ppl_B"]
    stats = make_zero_stats(*STAT_KEYS)

    logger.info("Starting training (363-dim + FK + CUT)...")
    for nb_iter in range(args.total_iter):
        w = get_weights(nb_iter, args.warm_up_iter)

        # ---- Load ----
        motion_A, mask_A = next(iter_A)
        motion_B, mask_B = next(iter_B)
        motion_A = motion_A.cuda().float()
        motion_B = motion_B.cuda().float()
        mask_A = mask_A.cuda().float()
        mask_B = mask_B.cuda().float()

        # ---- Reconstruction ----
        recon_A, commit_A, ppl_A = net.forward_A(motion_A)
        recon_B, commit_B, ppl_B = net.forward_B(motion_B)

        l_recon_A = recon_loss(recon_A, motion_A, mask_A)
        l_recon_B = recon_loss(recon_B, motion_B, mask_B)
        l_vel_A = recon_loss.velocity(recon_A, motion_A, mask_A)
        l_vel_B = recon_loss.velocity(recon_B, motion_B, mask_B)

        # ---- Cycle ----
        l_cyc_ABA = torch.zeros(1, device="cuda")
        l_cyc_BAB = torch.zeros(1, device="cuda")
        if w["cycle"] > 0:
            l_cyc_ABA = recon_loss(net.cycle_ABA(motion_A), motion_A, mask_A)
            l_cyc_BAB = recon_loss(net.cycle_BAB(motion_B), motion_B, mask_B)

        # ---- Paired contrastive ----
        l_contrast = torch.zeros(1, device="cuda")
        if w["contrast"] > 0:
            p_A, p_B, p_mask = next(iter_paired)
            p_A = p_A.cuda().float()
            p_B = p_B.cuda().float()
            l_contrast = contrastive_loss(net.get_motion_emb_A(p_A), net.get_motion_emb_B(p_B))

        # ---- CUT unpaired contrastive (P0) ----
        l_unp_contrast = torch.zeros(1, device="cuda")
        if w["unp_contrast"] > 0:
            B_hat, _, _ = net.forward_AB(motion_A)
            feat_A = net._encode_feat(motion_A, net.encoder_A)
            feat_Bt = net._encode_feat(B_hat, net.encoder_B)
            l_unp_contrast = unp_contrast_loss(feat_A, feat_Bt)

        # ---- FK physical losses (P0) ----
        l_fk_sup = torch.zeros(1, device="cuda")
        l_fk_sc = torch.zeros(1, device="cuda")
        l_fk_ee = torch.zeros(1, device="cuda")
        l_contact = torch.zeros(1, device="cuda")
        l_bone_len = torch.zeros(1, device="cuda")

        if w["fk_sup"] > 0 or w["fk_sc"] > 0:
            fk_losses = fk_loss_fn(recon_B, motion_B, mask_B)
            l_fk_sup = fk_losses.get("fk_sup", l_fk_sup)
            l_fk_sc = fk_losses.get("fk_sc", l_fk_sc)
            l_fk_ee = fk_losses.get("fk_ee", l_fk_ee)
            l_contact = fk_losses.get("contact", l_contact)
            l_bone_len = fk_losses.get("bone_len", l_bone_len)

        # ---- Total loss ----
        loss = sum([
            w["recon"]    * (l_recon_A + l_recon_B),
            w["vel"]      * (l_vel_A + l_vel_B),
            w["commit"]   * (commit_A + commit_B),
            w["cycle"]    * (l_cyc_ABA + l_cyc_BAB),
            w["contrast"] * l_contrast,
            w["unp_contrast"] * l_unp_contrast,
            w["fk_sup"]   * l_fk_sup,
            w["fk_sc"]    * l_fk_sc,
            w["fk_ee"]    * l_fk_ee,
            w["contact"]  * l_contact,
            w["bone_len"] * l_bone_len,
        ])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ---- Accumulate stats ----
        for k, v in [("recon_A",l_recon_A),("recon_B",l_recon_B),("vel_A",l_vel_A),("vel_B",l_vel_B),
                     ("cyc_ABA",l_cyc_ABA),("cyc_BAB",l_cyc_BAB),("contrast",l_contrast),
                     ("unp_contrast",l_unp_contrast),("fk_sup",l_fk_sup),("fk_sc",l_fk_sc),
                     ("fk_ee",l_fk_ee),("contact",l_contact),("bone_len",l_bone_len),
                     ("commit",commit_A+commit_B),("ppl_A",ppl_A),("ppl_B",ppl_B)]:
            stats[k] += v.item()

        # ---- Logging ----
        if nb_iter % args.print_iter == 0:
            n = args.print_iter
            gs = args.warm_up_iter + nb_iter

            for k in stats:
                writer.add_scalar("Train/" + k, stats[k] / n, gs)
            wandb.log(dict(("Train/" + k, stats[k] / n) for k in stats), step=gs)

            ra, rb = stats["recon_A"]/n, stats["recon_B"]/n
            va, vb = stats["vel_A"]/n, stats["vel_B"]/n
            ca, cb = stats["cyc_ABA"]/n, stats["cyc_BAB"]/n
            ct, uc = stats["contrast"]/n, stats["unp_contrast"]/n
            fs, fe, fc = stats["fk_sup"]/n, stats["fk_ee"]/n, stats["contact"]/n
            pa, pb = stats["ppl_A"]/n, stats["ppl_B"]/n
            logger.info(
                "Iter %6d | ReconA=%.4f ReconB=%.4f | VelA=%.4f VelB=%.4f | "
                "Cyc=%.4f/%.4f | Cont=%.4f UnpC=%.4f | FK=%.4f EE=%.4f Ct=%.4f | PPL=%.1f/%.1f",
                nb_iter, ra, rb, va, vb, ca, cb, ct, uc, fs, fe, fc, pa, pb)

            # Physics metrics (P1)
            with torch.no_grad():
                phy = PhysicsMetrics.compute(recon_B, motion_B, mask_B)
                wandb.log(dict(("Physics/" + k, v) for k, v in phy.items()), step=gs)
                for k, v in phy.items():
                    writer.add_scalar("Physics/" + k, v, gs)

            stats = make_zero_stats(*STAT_KEYS)

        # ---- Checkpoint ----
        if nb_iter % args.save_iter == 0:
            ckpt = os.path.join(args.out_dir, "net_iter%07d.pth" % nb_iter)
            torch.save(dict(net=net.state_dict(), iter=nb_iter), ckpt)
            logger.info("Checkpoint -> %s", ckpt)

    logger.info("Training complete!")