"""utils/unpaired_contrastive.py - CUT-style unpaired contrastive loss

Key insight: translation itself provides supervision.
Take unpaired motion_A, translate to B via shared codebook,
then contrast encoder features at corresponding temporal positions.
This does NOT require paired A-B data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnpairedContrastiveLoss(nn.Module):
    """
    CUT-style temporal patch contrastive loss.
    
    Flow: motion_A -> Enc_A -> VQ -> Dec_B -> motion_B_hat
          Then contrast Enc_A(motion_A)[t] with Enc_B(motion_B_hat)[t]
          Positive: same temporal position t
          Negative: different temporal positions s != t
    """
    def __init__(self, temperature=0.07, num_patches=8):
        super().__init__()
        self.temperature = temperature
        self.num_patches = num_patches

    def forward(self, feat_A, feat_B_translated):
        """
        feat_A:           Encoder_A output features (B, T, C)
        feat_B_translated: Encoder_B(B_hat) output features (B, T, C)
        
        Returns: scalar loss
        """
        B, T, C = feat_A.shape
        device = feat_A.device

        # Randomly sample temporal patches to reduce computation
        if T > self.num_patches:
            idx = torch.randperm(T, device=device)[:self.num_patches]
            idx = idx.sort()[0]  # keep temporal order for consistency
            feat_A = feat_A[:, idx, :]
            feat_B_translated = feat_B_translated[:, idx, :]

        P = feat_A.shape[1]  # num patches

        # L2 normalize
        feat_A = F.normalize(feat_A, dim=-1)
        feat_B_translated = F.normalize(feat_B_translated, dim=-1)

        # Flatten batch and patch dims: (B*P, C)
        q = feat_A.reshape(B * P, C)
        k = feat_B_translated.reshape(B * P, C)

        # Similarity matrix: (B*P, B*P)
        sim = torch.matmul(q, k.T) / self.temperature

        # Diagonal = same temporal position = positive pairs
        # Off-diagonal = different positions = negative pairs
        labels = torch.arange(B * P, device=device)

        # Bidirectional InfoNCE
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

        return loss


class UnpairedContrastiveLossSimple(nn.Module):
    """
    Simpler version: contrast global motion embeddings.
    For each sample in batch, treat A->B translation as positive pair.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_A, emb_B_translated):
        """
        emb_A: (B, C) global embedding of original A motion
        emb_B_translated: (B, C) global embedding of translated B motion
        """
        emb_A = F.normalize(emb_A, dim=-1)
        emb_B_translated = F.normalize(emb_B_translated, dim=-1)

        sim = torch.matmul(emb_A, emb_B_translated.T) / self.temperature
        labels = torch.arange(emb_A.shape[0], device=emb_A.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss