"""
train_vq_v3.py - Cross-Embodiment VQ-VAE with P0+P1+P2 improvements

P0: 363-dim G1 + FK physical losses + CUT unpaired contrastive
P1: Physics metrics monitoring
P2: Orthogonal regularization on 6D rotations (exp_v4)

Usage:
  python train_vq_v3.py --data_dir_A ./dataset/smpl/new_joint_vecs 
      --data_dir_B ./dataset/unitreeg1/new_joint_vecs_v4 
      --stat_dir_A ./dataset/smpl/meta --stat_dir_B ./dataset/unitreeg1/meta_v4 
      --split_txt_dir ./dataset/splits --exp-name exp_v4
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.utils_model as utils_model
import options.option_vq as option_vq
from utils.cross_losses import MotionReconLoss, ContrastiveLoss
from utils.physics_losses import FKPositionLoss, PhysicsMetrics, G1_363_SLICES, G1_N_JOINTS, orthogonal_regularization
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
    """Stage-based curriculum learning.
    
    Stage 1 (0-5k):       reconstruction only
    Stage 2 (5k-10k):     + cycle + CUT unpaired contrastive
    Stage 3 (10k+):       + FK physical losses + paired contrastive + orthogonal regularization
    """
    if nb_iter < warm_up_iter:
        return dict(recon=1.0, vel=0.5, commit=0.02, cycle=0.0, contrast=0.0,
                    unp_contrast=0.0, fk_sup=0.0, fk_sc=0.0, fk_ee=0.0,
                    contact=0.0, bone_len=0.0, ortho_reg=0.0)
    elif nb_iter < warm_up_iter * 2:
        return dict(recon=1.0, vel=0.5, commit=0.02, cycle=0.1, contrast=0.0,
                    unp_contrast=0.05, fk_sup=0.0, fk_sc=0.0, fk_ee=0.0,
                    contact=0.0, bone_len=0.0, ortho_reg=0.0)
    else:
        # P2: ortho_reg=2.0 pushes decoder to output valid rotation matrices
        return dict(recon=1.0, vel=0.5, commit=0.02, cycle=0.1, contrast=0.05,
                    unp_contrast=0.05, fk_sup=0.8, fk_sc=0.3, fk_ee=1.5,
                    contact=0.3, bone_len=0.1, ortho_reg=2.0)


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

    train_loader_A = DataLoaderA(args.data_dir_A, train_txt, args.stat_dir_A, args.batch_size, args.window_size)
    train_loader_B = DataLoaderB(args.data_dir_B, train_txt, args.stat_dir_B, args.batch_size, args.window_size)
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
        quantizer_type=args.quantizer, mu=args.mu,
    ).cuda()

    start_iter = 0
    if args.resume_pth is not None and os.path.exists(args.resume_pth):
        logger.info("Resuming from %s", args.resume_pth)
        ckpt = torch.load(args.resume_pth, map_location="cuda")
        net.load_state_dict(ckpt["net"])
        start_iter = ckpt.get("iter", 0)
        logger.info("Resumed at iter %d", start_iter)

    # ========== 4. Losses ==========
    recon_loss = MotionReconLoss(loss_type="l1_smooth")
    contrastive_loss = ContrastiveLoss(temperature=0.1)
    unp_contrast_loss = UnpairedContrastiveLoss(temperature=0.1)
    fk_loss_fn = FKPositionLoss(supervised_weight=0.8, self_consist_weight=0.3, ee_weight=1.5).cuda()

    # ========== 5. Optimizer ==========
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.resume_pth is not None and os.path.exists(args.resume_pth):
        ckpt = torch.load(args.resume_pth, map_location="cuda")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

    STAT_KEYS = ["recon_A","recon_B","vel_A","vel_B","cyc_ABA","cyc_BAB",
                 "contrast","unp_contrast","fk_sup","fk_sc","fk_ee",
                 "contact","bone_len","ortho_reg","commit","ppl_A","ppl_B"]
    stats = make_zero_stats(*STAT_KEYS)

    # ========== 6. Training Loop ==========
    logger.info("Starting training (363-dim + FK + CUT + OrthoReg)...")
    nb_iter = start_iter
    global_step = args.warm_up_iter + start_iter

    while nb_iter < args.total_iter:
        nb_iter += 1
        w = get_weights(global_step, args.warm_up_iter)

        # ---- Warm-up LR ----
        if nb_iter <= args.warm_up_iter and start_iter == 0:
            _, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

        # ---- Load data ----
        motion_A, mask_A = next(iter_A)
        motion_A = motion_A.cuda().float()
        mask_A = mask_A.cuda()

        motion_B, mask_B = next(iter_B)
        motion_B = motion_B.cuda().float()
        mask_B = mask_B.cuda()

        # ---- Forward ----
        recon_A, recon_B, commit_A, commit_B, ppl_A, ppl_B = net(motion_A, motion_B)

        # ---- Reconstruction ----
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

        # ---- CUT unpaired contrastive ----
        l_unp_contrast = torch.zeros(1, device="cuda")
        if w["unp_contrast"] > 0:
            B_hat, _, _ = net.forward_AB(motion_A)
            feat_A = net._encode_feat(motion_A, net.encoder_A)
            feat_Bt = net._encode_feat(B_hat, net.encoder_B)
            l_unp_contrast = unp_contrast_loss(feat_A, feat_Bt)

        # ---- FK physical losses ----
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

        # ---- Orthogonal regularization (P2) ----
        l_ortho_reg = torch.zeros(1, device="cuda")
        if w["ortho_reg"] > 0:
            rs, re = G1_363_SLICES["rot_data"]
            pred_rot = recon_B[..., rs:re].reshape(recon_B.shape[0], recon_B.shape[1], G1_N_JOINTS, 6)
            l_ortho_reg = orthogonal_regularization(pred_rot, mask_B)

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
            w["ortho_reg"] * l_ortho_reg,
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
                     ("ortho_reg",l_ortho_reg),
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
            oreg = stats["ortho_reg"]/n
            pa, pb = stats["ppl_A"]/n, stats["ppl_B"]/n
            logger.info(
                "Iter %6d | ReconA=%.4f ReconB=%.4f | VelA=%.4f VelB=%.4f | "
                "Cyc=%.4f/%.4f | Cont=%.4f UnpC=%.4f | FK=%.4f EE=%.4f Ct=%.4f Ortho=%.4f | PPL=%.1f/%.1f",
                nb_iter, ra, rb, va, vb, ca, cb, ct, uc, fs, fe, fc, oreg, pa, pb)

            # Physics metrics
            with torch.no_grad():
                phy = PhysicsMetrics.compute(recon_B, motion_B, mask_B)
                wandb.log(dict(("Physics/" + k, v) for k, v in phy.items()), step=gs)
                for k, v in phy.items():
                    writer.add_scalar("Physics/" + k, v, gs)

            stats = make_zero_stats(*STAT_KEYS)

        # ---- Checkpoint ----
        if nb_iter % args.save_iter == 0:
            ckpt = os.path.join(args.out_dir, "net_iter%07d.pth" % nb_iter)
            torch.save(dict(net=net.state_dict(), iter=nb_iter, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), ckpt)
            logger.info("Checkpoint -> %s", ckpt)

        global_step += 1

    logger.info("Training complete!")