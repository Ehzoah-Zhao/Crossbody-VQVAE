'''
train_vq_v2.py --- 跨本体 VQ-VAE 训练脚本 (修复版)

核心改进 vs v1:
  1. 加入直接的 cross-embodiment 监督损失 w_cross_direct (默认 1.0)
  2. 合理的权重调度: warm-up phase -> ramp-up phase -> full training
  3. 周期性验证集评估 cross-embodiment MSE
  4. 完善的 WandB + TensorBoard 日志

训练三阶段:
  Phase 1 (warm-up, 0 ~ warm_up_iter):
    仅自重建, cross=0, cycle=0, contrast=0
  Phase 2 (ramp-up, warm_up_iter ~ warm_up_iter + cross_ramp_iter):
    cross: 0 -> full, cycle: 0 -> 0.1, contrast: 0 -> 0.05
  Phase 3 (full training):
    所有权重全量激活
'''

import os, json, warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.utils_model as utils_model
import options.option_vq as option_vq
from utils.cross_losses import MotionReconLoss, ContrastiveLoss
from dataset.dataset_A import DATALoader as DataLoaderA
from dataset.dataset_B import DATALoader as DataLoaderB
from dataset.paired_dataset import PairedDATALoader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    return optimizer, current_lr


def make_zero_stats(*keys):
    return {k: 0. for k in keys}


if __name__ == '__main__':
    # ================= 1. 配置 =================
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    import wandb
    resume_wandb = args.resume_pth is not None
    wandb_id = None
    if resume_wandb:
        ckpt_tmp = torch.load(args.resume_pth, map_location='cpu', weights_only=False)
        wandb_id = ckpt_tmp.get('wandb_id', None)
    wandb.init(project='Crossbody-VQVAE', name=args.exp_name, config=vars(args),
               id=wandb_id, resume='must' if wandb_id else None)

    # ================= 2. 数据加载 =================
    train_txt = os.path.join(args.split_txt_dir, 'train.txt')
    val_txt   = os.path.join(args.split_txt_dir, 'val.txt')

    logger.info('Loading datasets...')
    train_loader_A = DataLoaderA(args.data_dir_A, train_txt, args.stat_dir_A,
                                 args.batch_size, args.window_size)
    train_loader_B = DataLoaderB(args.data_dir_B, train_txt, args.stat_dir_B,
                                 args.batch_size, args.window_size)
    paired_loader = PairedDATALoader(args.data_dir_A, args.data_dir_B, train_txt,
                                     args.stat_dir_A, args.stat_dir_B,
                                     args.batch_size_paired, args.window_size)
    val_loader = PairedDATALoader(args.data_dir_A, args.data_dir_B, val_txt,
                                  args.stat_dir_A, args.stat_dir_B,
                                  args.val_samples, args.window_size)

    iter_A = cycle(train_loader_A)
    iter_B = cycle(train_loader_B)
    iter_paired = cycle(paired_loader)

    # 取一批验证数据
    val_batch = next(iter(val_loader))
    v_A, v_B, v_mask = [x.cuda().float() for x in val_batch]
    logger.info(f'Val set: {v_A.shape[0]} samples, window={v_A.shape[1]}')

    # ================= 3. 模型 =================
    net = vqvae.CrossEmbodimentVQVAE(
        input_dim_A=args.input_dim_A,
        input_dim_B=args.input_dim_B,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm,
        quantizer_type=args.quantizer,
        mu=args.mu,
    ).cuda()

    start_iter = 0
    if args.resume_pth:
        logger.info(f'Resuming from {args.resume_pth}')
        ckpt = torch.load(args.resume_pth, map_location='cuda')
        net.load_state_dict(ckpt['net'], strict=True)
        if 'iter' in ckpt:
            start_iter = ckpt['iter']
            logger.info(f'Resuming at iteration {start_iter}')
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info('Optimizer state restored')
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info('Scheduler state restored')

    # ================= 4. 优化器 =================
    optimizer = optim.AdamW(net.parameters(), lr=args.lr,
                            betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=args.lr_scheduler,
                                                gamma=args.gamma)

    recon_loss = MotionReconLoss(loss_type=args.recons_loss)
    contrastive_loss_fn = ContrastiveLoss(temperature=args.temperature)

    STAT_KEYS = ['recon_A', 'recon_B', 'vel_A', 'vel_B',
                 'cross_AB', 'cross_BA', 'cycle_ABA', 'cycle_BAB',
                 'contrast', 'commit', 'ppl_A', 'ppl_B']

    # ================= 5. 权重调度 =================
    def get_weights(iter_num):
        w = {
            'recon':    args.w_recon,
            'vel':      args.w_vel,
            'commit':   args.w_commit,
            'cross':    0.0,
            'cycle':    0.0,
            'contrast': 0.0,
        }
        if iter_num > args.warm_up_iter:
            cross_iters = iter_num - args.warm_up_iter
            ramp = min(1.0, cross_iters / args.cross_ramp_iter)
            w['cross']    = args.w_cross_direct * ramp
            w['cycle']    = 0.1 * ramp
            w['contrast'] = 0.05 * ramp
        return w

    # ================= 6. 训练主循环 =================
    if start_iter == 0:
        logger.info('=== Phase 1: Warm-up (self-recon only) ===')
    else:
        logger.info(f'=== Resumed at iter {start_iter}, continuing ===')
    stats = make_zero_stats(*STAT_KEYS)

    for nb_iter in range(start_iter + 1, args.total_iter + 1):
        w = get_weights(nb_iter)

        # --- 获取数据 ---
        motion_A, mask_A = next(iter_A)
        motion_B, mask_B = next(iter_B)
        motion_A = motion_A.cuda().float(); mask_A = mask_A.cuda().float()
        motion_B = motion_B.cuda().float(); mask_B = mask_B.cuda().float()

        # --- (a) 自重建 ---
        recon_A, commit_A, ppl_A = net.forward_A(motion_A)
        recon_B, commit_B, ppl_B = net.forward_B(motion_B)

        l_recon_A = recon_loss(recon_A, motion_A, mask_A)
        l_recon_B = recon_loss(recon_B, motion_B, mask_B)
        l_vel_A   = recon_loss.velocity(recon_A, motion_A, mask_A)
        l_vel_B   = recon_loss.velocity(recon_B, motion_B, mask_B)

        # --- (b) 直接跨本体监督 【核心新增】 ---
        l_cross_AB = torch.zeros(1, device='cuda')
        l_cross_BA = torch.zeros(1, device='cuda')
        if w['cross'] > 0:
            p_A, p_B, p_mask = next(iter_paired)
            p_A = p_A.cuda().float(); p_B = p_B.cuda().float()
            p_mask = p_mask.cuda().float()

            cross_B, _, _ = net.forward_AB(p_A)
            cross_A, _, _ = net.forward_BA(p_B)

            l_cross_AB = recon_loss(cross_B, p_B, p_mask)
            l_cross_BA = recon_loss(cross_A, p_A, p_mask)

        # --- (c) 循环一致性 ---
        l_cycle_ABA = torch.zeros(1, device='cuda')
        l_cycle_BAB = torch.zeros(1, device='cuda')
        if w['cycle'] > 0:
            A_cycle = net.cycle_ABA(motion_A)
            B_cycle = net.cycle_BAB(motion_B)
            l_cycle_ABA = recon_loss(A_cycle, motion_A, mask_A)
            l_cycle_BAB = recon_loss(B_cycle, motion_B, mask_B)

        # --- (d) 对比损失 ---
        l_contrast = torch.zeros(1, device='cuda')
        if w['contrast'] > 0:
            p_A, p_B, p_mask = next(iter_paired)
            p_A = p_A.cuda().float(); p_B = p_B.cuda().float()
            emb_A = net.get_motion_emb_A(p_A)
            emb_B = net.get_motion_emb_B(p_B)
            l_contrast = contrastive_loss_fn(emb_A, emb_B)

        # --- 总损失 ---
        loss = (w['recon']    * (l_recon_A + l_recon_B)
              + w['vel']      * (l_vel_A + l_vel_B)
              + w['commit']   * (commit_A + commit_B)
              + w['cross']    * (l_cross_AB + l_cross_BA)
              + w['cycle']    * (l_cycle_ABA + l_cycle_BAB)
              + w['contrast'] * l_contrast)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        if nb_iter <= args.warm_up_iter and start_iter < args.warm_up_iter:
            optimizer, _ = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        elif nb_iter > args.warm_up_iter:
            scheduler.step()

        # --- 累计统计 ---
        stats['recon_A']   += l_recon_A.item()
        stats['recon_B']   += l_recon_B.item()
        stats['vel_A']     += l_vel_A.item()
        stats['vel_B']     += l_vel_B.item()
        stats['cross_AB']  += l_cross_AB.item()
        stats['cross_BA']  += l_cross_BA.item()
        stats['cycle_ABA'] += l_cycle_ABA.item()
        stats['cycle_BAB'] += l_cycle_BAB.item()
        stats['contrast']  += l_contrast.item()
        stats['commit']    += (commit_A + commit_B).item()
        stats['ppl_A']     += ppl_A.item()
        stats['ppl_B']     += ppl_B.item()

        # --- 日志与验证 ---
        if nb_iter % args.print_iter == 0:
            n = args.print_iter
            global_step = nb_iter  # 使用实际 iter 数

            # 周期性验证
            if nb_iter % args.val_interval == 0 and w['cross'] > 0:
                net.eval()
                with torch.no_grad():
                    vcB, _, _ = net.forward_AB(v_A)
                    vcA, _, _ = net.forward_BA(v_B)
                    stats['val_cross_AB_tmp'] = recon_loss(vcB, v_B, v_mask).item()
                    stats['val_cross_BA_tmp'] = recon_loss(vcA, v_A, v_mask).item()
                net.train()
                writer.add_scalar('Val/cross_AB', stats['val_cross_AB_tmp'], nb_iter)
                writer.add_scalar('Val/cross_BA', stats['val_cross_BA_tmp'], nb_iter)
                wandb.log({'Val/cross_AB': stats['val_cross_AB_tmp'],
                           'Val/cross_BA': stats['val_cross_BA_tmp']}, step=nb_iter)
                logger.info(f'  >>> VAL | cross_AB={stats["val_cross_AB_tmp"]:.6f}  cross_BA={stats["val_cross_BA_tmp"]:.6f}')

            for k in stats:
                writer.add_scalar(f'Train/{k}', stats[k] / n, global_step)
            wandb.log({f'Train/{k}': stats[k] / n for k in stats}, step=global_step)

            logger.info(
                f'Iter {nb_iter:6d} | '
                f'W(cross={w["cross"]:.2f} cyc={w["cycle"]:.2f} cont={w["contrast"]:.2f}) | '
                f'RecA={stats["recon_A"]/n:.4f} RecB={stats["recon_B"]/n:.4f} | '
                f'CrossAB={stats["cross_AB"]/n:.4f} CrossBA={stats["cross_BA"]/n:.4f} | '
                f'Cyc={stats["cycle_ABA"]/n:.4f}/{stats["cycle_BAB"]/n:.4f} | '
                f'Cont={stats["contrast"]/n:.4f} | Cmt={stats["commit"]/n:.4f}'
            )
            stats = make_zero_stats(*STAT_KEYS)

        # --- 保存 checkpoint ---
        if nb_iter % args.save_iter == 0:
            ckpt_path = os.path.join(args.out_dir, f'net_iter{nb_iter:07d}.pth')
            torch.save({
                'net': net.state_dict(),
                'iter': nb_iter,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'wandb_id': wandb.run.id,
            }, ckpt_path)
            logger.info(f'Checkpoint saved -> {ckpt_path}')

    logger.info('=== Training completed ===')
