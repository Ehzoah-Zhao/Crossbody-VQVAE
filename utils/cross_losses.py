"""
utils/cross_losses.py
跨本体 VQ-VAE 的 Loss 函数集合

包含:
  - MotionReconLoss : 逐帧重建 + 速度平滑 (支持 Mask 掩码)
  - ContrastiveLoss : InfoNCE 配对对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionReconLoss(nn.Module):
    """
    逐帧重建损失，支持 Mask 掩码机制。
    只计算有效帧的重建误差，忽略 Padding 的 0。
    """
    def __init__(self, loss_type: str = 'l1_smooth'):
        super().__init__()
        # 注意：这里必须设置 reduction='none'，否则 PyTorch 会自动求均值，我们就没法乘 Mask 了
        _map = {
            'l1':        nn.L1Loss(reduction='none'),
            'l2':        nn.MSELoss(reduction='none'),
            'l1_smooth': nn.SmoothL1Loss(reduction='none'),
        }
        if loss_type not in _map:
            raise ValueError(f"未知 loss_type: {loss_type}，可选: {list(_map)}")
        self.loss_fn  = _map[loss_type]

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: (B, T, C)
        mask: (B, T)
        """
        loss = self.loss_fn(pred, gt)              # 输出 shape: (B, T, C)
        mask_expanded = mask.unsqueeze(-1)         # 扩展 shape: (B, T, 1) 以便广播相乘
        
        loss_masked = loss * mask_expanded
        
        # 只对有效帧求平均 (分母加上 1e-8 防止除以 0)
        return loss_masked.sum() / (mask_expanded.sum() * gt.shape[-1] + 1e-8)

    def velocity(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        帧间差分损失，惩罚抖动，同样支持 Mask。
        """
        pred_vel = pred[:, 1:] - pred[:, :-1]
        gt_vel   = gt[:, 1:]   - gt[:, :-1]
        
        loss_vel = self.loss_fn(pred_vel, gt_vel)  # 输出 shape: (B, T-1, C)
        
        # 速度掩码比原掩码少一帧：只有相邻两帧都有效(1 * 1 = 1)时，计算的速度才有效
        mask_vel = mask[:, 1:] * mask[:, :-1]      
        mask_vel_expanded = mask_vel.unsqueeze(-1)
        
        loss_vel_masked = loss_vel * mask_vel_expanded
        return loss_vel_masked.sum() / (mask_vel_expanded.sum() * gt.shape[-1] + 1e-8)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE 对比损失，用于配对动作的隐空间语义对齐。
    正样本对: 同一语义动作在 A 和 B 中的全局嵌入
    负样本:   batch 内其他动作
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_A: torch.Tensor, emb_B: torch.Tensor) -> torch.Tensor:
        """
        emb_A, emb_B: (B, C)
        要求 batch 内每条 emb_A[i] 与 emb_B[i] 是正样本对。
        """
        # L2 归一化到单位球面
        emb_A = F.normalize(emb_A, dim=-1)
        emb_B = F.normalize(emb_B, dim=-1)

        # (B, B) 相似度矩阵，对角线是正样本
        sim = torch.matmul(emb_A, emb_B.T) / self.temperature

        labels = torch.arange(emb_A.shape[0], device=emb_A.device)

        # 双向 InfoNCE：A→B 和 B→A 各算一次再平均
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss