"""models/urdf_encoder.py - Embodiment embedding for multi-robot support

Level 1: Simple per-robot learnable embedding (current)
Level 2: URDF graph encoder (future, for diverse robots)
"""

import torch
import torch.nn as nn


class EmbodimentEmbedding(nn.Module):
    """
    Level 1: Simple per-robot learnable embedding.
    Each robot gets a unique ID -> learnable embedding vector.
    
    Usage:
        emb = EmbodimentEmbedding(num_robots=2, emb_dim=256)
        emb_A = emb(robot_id=0)  # (B,) -> (B, 256)
        emb_B = emb(robot_id=1)
    """
    def __init__(self, num_robots=2, emb_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_robots, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, robot_id):
        """
        robot_id: int or LongTensor of shape (B,)
        Returns: (B, emb_dim)
        """
        if isinstance(robot_id, int):
            device = self.embedding.weight.device
            robot_id = torch.tensor([robot_id], device=device, dtype=torch.long)
        return self.embedding(robot_id)


class EmbodimentDecoderWrapper(nn.Module):
    """
    Wraps decoder to accept embodiment embedding as conditioning.
    Concatenates embodiment embedding to decoder input.
    """
    def __init__(self, decoder, emb_dim=256):
        super().__init__()
        self.decoder = decoder
        self.emb_proj = nn.Linear(emb_dim, decoder.output_emb_width if hasattr(decoder, "output_emb_width") else 512)

    def forward(self, x, emb):
        """
        x: (B, C, T) decoder input features
        emb: (B, emb_dim) embodiment embedding
        Returns: decoder output
        """
        emb_proj = self.emb_proj(emb).unsqueeze(-1)  # (B, C, 1)
        x_cond = x + emb_proj.expand(-1, -1, x.shape[-1])
        return self.decoder(x_cond)