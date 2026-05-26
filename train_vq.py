"""
train_vq.py  —  跨本体 VQ-VAE 训练脚本（规范化 Windows 多进程兼容版）
集成了 Mask 掩码机制，剔除了零填充对 Codebook 的污染。

训练阶段建议（课程学习）:
  阶段 1  warm-up:   仅自重建，让 A/B 各自 encoder-decoder 收敛
  阶段 2  main:      加入 cycle loss（w_cycle 从 0 → 0.1）
  阶段 3  main:      加入 contrastive loss（w_contrast 从 0 → 0.05，需配对数据）
"""

import os
import json
import warnings
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


# ─────────────────────────────────────────────────────────────────────
# 工具函数（留在全局作用域）
# ─────────────────────────────────────────────────────────────────────

def cycle(iterable):
    """将有限的 DataLoader 包装成无限迭代器"""
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


# ─────────────────────────────────────────────────────────────────────
# 主训练入口（通过入口保护解决 Windows 系统的 spawn 报错）
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    
    # ── 1. 初始化配置 ────────────────────────────────────────────────
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # ── 2. 数据集加载（对齐包含划分配对与归一化参数的新接口） ─────────────
    # 组合训练集划分名单 train.txt 的完整路径
    train_txt_path = os.path.join(args.split_txt_dir, 'train.txt')
    
    logger.info("Loading Datasets...")
    train_loader_A = DataLoaderA(
        args.data_dir_A, 
        train_txt_path, 
        args.stat_dir_A, 
        args.batch_size, 
        args.window_size
    )
    train_loader_B = DataLoaderB(
        args.data_dir_B, 
        train_txt_path, 
        args.stat_dir_B, 
        args.batch_size, 
        args.window_size
    )
    train_loader_paired = PairedDATALoader(
        args.data_dir_A, 
        args.data_dir_B, 
        train_txt_path, 
        args.stat_dir_A, 
        args.stat_dir_B, 
        args.batch_size_paired, 
        args.window_size
    )

    # 包装为无限迭代器
    iter_A = cycle(train_loader_A)
    iter_B = cycle(train_loader_B)
    iter_paired = cycle(train_loader_paired)

    # ── 3. 模型构建 ──────────────────────────────────────────────────
    net = vqvae.CrossEmbodimentVQVAE(
        input_dim_A          = args.input_dim_A,       # SMPL: 134
        input_dim_B          = args.input_dim_B,       # G1: 186
        nb_code              = args.nb_code,
        code_dim             = args.code_dim,
        output_emb_width     = args.output_emb_width,
        down_t               = args.down_t,
        stride_t             = args.stride_t,
        width                = args.width,
        depth                = args.depth,
        dilation_growth_rate = args.dilation_growth_rate,
        activation           = args.vq_act,
        norm                 = args.vq_norm,
        quantizer_type       = args.quantizer,
        mu                   = args.mu,                
    )

    if args.resume_pth:
        logger.info(f'Loading checkpoint from {args.resume_pth}')
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)

    net.train()
    net.cuda()

    # ── 4. 优化器 & 调度器 ────────────────────────────────────────────
    optimizer = optim.AdamW(net.parameters(), lr=args.lr,
                            betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    # ── 5. Loss 函数 & 权重映射 ────────────────────────────────────────
    recon_loss       = MotionReconLoss(args.recons_loss)
    contrastive_loss = ContrastiveLoss(temperature=args.temperature)  

    w = {
        'recon':    args.w_recon,      # 默认: 1.0
        'commit':   args.w_commit,     # 默认: 0.02
        'vel':      args.w_vel,        # 默认: 0.1
        'cycle':    args.w_cycle,      # 课程学习控制开关：先 0.0 → 后 0.1
        'contrast': args.w_contrast,   # 课程学习控制开关：先 0.0 → 后 0.05
    }
    logger.info(f"Loss weights: {w}")

    # ── 6. Warm-up 阶段（仅自重建训练， w_cycle=0, w_contrast=0） ──────
    logger.info("=" * 60)
    logger.info("Warm-up phase: 自重建训练 (已接入 Mask 机制)")
    logger.info("=" * 60)

    STAT_KEYS_WU = ['recon_A', 'recon_B', 'vel_A', 'vel_B', 'commit', 'ppl_A', 'ppl_B']
    stats = make_zero_stats(*STAT_KEYS_WU)

    for nb_iter in range(1, args.warm_up_iter + 1):
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

        # 接收真实的动作数据特征与对应的掩码 Mask
        motion_A, mask_A = next(iter_A)
        motion_B, mask_B = next(iter_B)

        motion_A, mask_A = motion_A.cuda().float(), mask_A.cuda().float()
        motion_B, mask_B = motion_B.cuda().float(), mask_B.cuda().float()

        # 前向传播自重建
        recon_A, commit_A, ppl_A = net.forward_A(motion_A)
        recon_B, commit_B, ppl_B = net.forward_B(motion_B)

        # 计算带 Mask 的重建误差与时序平滑度，避免零填充污染特征空间
        l_recon_A = recon_loss(recon_A, motion_A, mask_A)
        l_recon_B = recon_loss(recon_B, motion_B, mask_B)
        l_vel_A   = recon_loss.velocity(recon_A, motion_A, mask_A)
        l_vel_B   = recon_loss.velocity(recon_B, motion_B, mask_B)

        loss = (w['recon']  * (l_recon_A + l_recon_B)
              + w['vel']    * (l_vel_A   + l_vel_B)
              + w['commit'] * (commit_A  + commit_B))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标累加
        stats['recon_A'] += l_recon_A.item()
        stats['recon_B'] += l_recon_B.item()
        stats['vel_A']   += l_vel_A.item()
        stats['vel_B']   += l_vel_B.item()
        stats['commit']  += (commit_A + commit_B).item()
        stats['ppl_A']   += ppl_A.item()
        stats['ppl_B']   += ppl_B.item()

        if nb_iter % args.print_iter == 0:
            n = args.print_iter
            logger.info(
                f"Warmup [{nb_iter:6d}/{args.warm_up_iter}] lr={current_lr:.5f} | "
                f"ReconA={stats['recon_A']/n:.4f} ReconB={stats['recon_B']/n:.4f} | "
                f"VelA={stats['vel_A']/n:.4f} VelB={stats['vel_B']/n:.4f} | "
                f"Commit={stats['commit']/n:.4f} | "
                f"PPL_A={stats['ppl_A']/n:.1f} PPL_B={stats['ppl_B']/n:.1f}"
            )
            stats = make_zero_stats(*STAT_KEYS_WU)

    # ── 7. 主训练循环（自重建 + 循环一致性 + 全局对比损失对齐） ──────────
    logger.info("=" * 60)
    logger.info("Main training phase: 开启跨本体多任务协同对齐")
    logger.info("=" * 60)

    # ================= 【新增部分开始】 =================
    # 1. 在 Warm-up 结束时，立刻强制保存一个干净的基础模型
    ckpt_path = os.path.join(args.out_dir, 'net_warmup_done.pth')
    torch.save({'net': net.state_dict(), 'iter': args.warm_up_iter}, ckpt_path)
    logger.info(f"Warm-up Checkpoint saved successfully → {ckpt_path}")

    # 2. 自动修改权重，无缝开启跨本体对齐（解放双手）
    w['cycle'] = 0.05
    w['contrast'] = 0.05
    logger.info(f"Auto-updated Loss weights for Main phase: {w}")
    # ================= 【新增部分结束】 =================

    STAT_KEYS = ['recon_A', 'recon_B', 'vel_A', 'vel_B',
                 'cycle_ABA', 'cycle_BAB', 'contrast', 'commit', 'ppl_A', 'ppl_B']
    stats = make_zero_stats(*STAT_KEYS)

    for nb_iter in range(1, args.total_iter + 1):

        # 独立获取非配对（或全量）训练集的动作及掩码
        motion_A, mask_A = next(iter_A)
        motion_B, mask_B = next(iter_B)

        motion_A, mask_A = motion_A.cuda().float(), mask_A.cuda().float()
        motion_B, mask_B = motion_B.cuda().float(), mask_B.cuda().float()

        # ── 7.1 自重建分支 ──
        recon_A, commit_A, ppl_A = net.forward_A(motion_A)
        recon_B, commit_B, ppl_B = net.forward_B(motion_B)

        l_recon_A = recon_loss(recon_A, motion_A, mask_A)
        l_recon_B = recon_loss(recon_B, motion_B, mask_B)
        l_vel_A   = recon_loss.velocity(recon_A, motion_A, mask_A)
        l_vel_B   = recon_loss.velocity(recon_B, motion_B, mask_B)

        # ── 7.2 循环一致性分支（通过隐空间直通梯度强迫 Decoder 覆盖全局） ──
        l_cycle_ABA = torch.zeros(1, device='cuda')
        l_cycle_BAB = torch.zeros(1, device='cuda')
        if w['cycle'] > 0:
            A_cycle = net.cycle_ABA(motion_A)   # A → B̂ → Â_cycle
            B_cycle = net.cycle_BAB(motion_B)   # B → Â → B̂_cycle
            
            # 循环恢复出的动作片段同样受对应的真实时序有效长度限制
            l_cycle_ABA = recon_loss(A_cycle, motion_A, mask_A)
            l_cycle_BAB = recon_loss(B_cycle, motion_B, mask_B)

        # ── 7.3 全局动作语义对比分支（强拉配对动作，强推负样本动作） ──
        l_contrast = torch.zeros(1, device='cuda')
        if w['contrast'] > 0:
            # 接收完全时间同步对齐的配对资产特征与共享的掩码
            p_A, p_B, p_mask = next(iter_paired)
            p_A, p_B, p_mask = p_A.cuda().float(), p_B.cuda().float(), p_mask.cuda().float()
            
            # 均值池化压缩为全局动作嵌入表示（1D-Conv提取特征）
            emb_A = net.get_motion_emb_A(p_A)   
            emb_B = net.get_motion_emb_B(p_B)   
            l_contrast = contrastive_loss(emb_A, emb_B)

        # ── 7.4 反向传播与梯度更新 ──
        loss = (w['recon']    * (l_recon_A + l_recon_B)
              + w['vel']      * (l_vel_A   + l_vel_B)
              + w['commit']   * (commit_A  + commit_B)
              + w['cycle']    * (l_cycle_ABA + l_cycle_BAB)
              + w['contrast'] * l_contrast)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 指标记录累加
        stats['recon_A']   += l_recon_A.item()
        stats['recon_B']   += l_recon_B.item()
        stats['vel_A']     += l_vel_A.item()
        stats['vel_B']     += l_vel_B.item()
        stats['cycle_ABA'] += l_cycle_ABA.item()
        stats['cycle_BAB'] += l_cycle_BAB.item()
        stats['contrast']  += l_contrast.item()
        stats['commit']    += (commit_A + commit_B).item()
        stats['ppl_A']     += ppl_A.item()
        stats['ppl_B']     += ppl_B.item()

        # ── 7.5 日志打印与 TensorBoard 监控 ──
        if nb_iter % args.print_iter == 0:
            n = args.print_iter
            for k in stats:
                writer.add_scalar(f'Train/{k}', stats[k] / n, nb_iter)

            logger.info(
                f"Iter {nb_iter:6d} | "
                f"ReconA={stats['recon_A']/n:.4f} ReconB={stats['recon_B']/n:.4f} | "
                f"VelA={stats['vel_A']/n:.4f} VelB={stats['vel_B']/n:.4f} | "
                f"CycABA={stats['cycle_ABA']/n:.4f} CycBAB={stats['cycle_BAB']/n:.4f} | "
                f"Contrast={stats['contrast']/n:.4f} | "
                f"Commit={stats['commit']/n:.4f} | "
                f"PPL_A={stats['ppl_A']/n:.1f} PPL_B={stats['ppl_B']/n:.1f}"
            )
            stats = make_zero_stats(*STAT_KEYS)

        # ── 7.6 周期性高频保存 Checkpoint（防止笔记本意外中断） ──
        if nb_iter % args.save_iter == 0:   
            ckpt_path = os.path.join(args.out_dir, f'net_iter{nb_iter:07d}.pth')
            torch.save({'net': net.state_dict(), 'iter': nb_iter}, ckpt_path)
            logger.info(f"Checkpoint saved successfully → {ckpt_path}")