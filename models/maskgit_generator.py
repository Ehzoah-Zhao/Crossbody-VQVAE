"""models/maskgit_generator.py - MaskGIT-style discrete diffusion generator

Operates on VQ token space (discrete indices from shared codebook).
Training: randomly mask tokens, bidirectional Transformer predicts masked tokens.
Inference: iterative unmasking from all-[MASK] state.

Architecture: Bidirectional Transformer (like BERT) with condition embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskGITGenerator(nn.Module):
    """
    MaskGIT-style generator on VQ discrete token space.
    
    Training:
        1. VQ-VAE encode motion -> token indices (B, T)
        2. Randomly mask some tokens
        3. Bidirectional Transformer predicts masked tokens
        4. Loss = cross-entropy on masked positions
    
    Inference:
        1. Start from all [MASK] tokens
        2. Iteratively predict and unmask (cosine schedule)
        3. Feed final tokens to frozen VQ-VAE decoder
    """
    def __init__(self, vocab_size=512, code_dim=512, max_len=64,
                 cond_dim=256, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size
        self.max_len = max_len
        self.code_dim = code_dim

        # Token + mask embedding
        self.token_embed = nn.Embedding(vocab_size + 1, code_dim)  # +1 for [MASK]

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, code_dim) * 0.02)

        # Condition projection (robot ID embedding -> code_dim)
        self.cond_proj = nn.Linear(cond_dim, code_dim)

        # Bidirectional Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=code_dim, nhead=num_heads, dim_feedforward=code_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        self.head = nn.Linear(code_dim, vocab_size)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, token_ids, condition):
        """
        token_ids: (B, T) integer tokens, masked positions filled with mask_token_id
        condition: (B, cond_dim) robot condition embedding
        Returns: (B, T, vocab_size) logits
        """
        B, T = token_ids.shape

        # Token + Position + Condition
        x = self.token_embed(token_ids)  # (B, T, D)
        x = x + self.pos_embed[:, :T, :]
        cond_emb = self.cond_proj(condition).unsqueeze(1)  # (B, 1, D)
        x = x + cond_emb

        # Bidirectional transformer
        x = self.transformer(x)  # (B, T, D)

        # Predict token logits
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    def compute_loss(self, token_ids, condition, mask_ratio=0.5):
        """
        Compute masked token prediction loss.
        
        token_ids: (B, T) ground truth token indices
        condition: (B, cond_dim)
        mask_ratio: fraction of tokens to mask
        """
        B, T = token_ids.shape
        device = token_ids.device

        # Randomly select positions to mask
        rand = torch.rand(B, T, device=device)
        mask = rand < mask_ratio

        # Create masked input
        masked_ids = token_ids.clone()
        masked_ids[mask] = self.mask_token_id

        # Forward
        logits = self.forward(masked_ids, condition)

        # Loss only on masked positions
        loss = F.cross_entropy(logits[mask], token_ids[mask], reduction="mean")
        return loss

    @torch.no_grad()
    def generate(self, condition, seq_len=None, steps=8, temperature=1.0):
        """
        Iterative decoding: all-[MASK] -> fully unmasked.
        
        condition: (B, cond_dim)
        seq_len: output sequence length (default max_len)
        steps: number of iterative refinement steps
        temperature: sampling temperature
        """
        B = condition.shape[0]
        T = seq_len if seq_len is not None else self.max_len
        device = condition.device

        # Start from all mask
        tokens = torch.full((B, T), self.mask_token_id, device=device, dtype=torch.long)

        for step in range(steps):
            # Cosine schedule: ratio of tokens still masked
            if steps > 1:
                ratio = np.cos(np.pi / 2 * step / steps)
            else:
                ratio = 0.0

            # Predict all tokens
            logits = self.forward(tokens, condition)
            probs = F.softmax(logits / temperature, dim=-1)

            # Sample from predicted distribution
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(B, T)

            # Decide which tokens to update
            is_masked = (tokens == self.mask_token_id)
            # Confidence = max probability for each position
            confidence = probs.max(dim=-1).values
            confidence[~is_masked] = -float("inf")

            # Number of tokens to unmask this step
            n_unmask = int(T * (1 - ratio)) - (~is_masked).sum(dim=1).float().mean().item()
            n_unmask = max(1, int(n_unmask))

            if n_unmask > 0:
                # Select top-k confident masked positions
                _, top_idx = confidence.topk(min(n_unmask, is_masked.sum().item()), dim=-1)
                for b in range(B):
                    tokens[b, top_idx[b]] = sampled[b, top_idx[b]]

        return tokens