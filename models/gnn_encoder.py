"""models/gnn_encoder.py - GNN encoder based on kinematic tree

Spatial-Temporal GNN: per-frame message passing along kinematic tree,
then temporal convolution for output compatible with existing VQ-VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KinematicGNNEncoder(nn.Module):
    """
    Spatial-Temporal GNN encoder.
    Input:  (B, T, J, in_dim) per-joint features per frame
    Output: (B, out_dim, T_out) compatible with existing Encoder output
    """
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=512,
                 num_joints=30, num_layers=3,
                 down_t=3, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation="relu"):
        super().__init__()
        self.num_joints = num_joints

        # Joint feature projection
        self.joint_proj = nn.Linear(in_dim, hidden_dim)

        # Message passing layers along kinematic tree
        self.gnn_layers = nn.ModuleList([
            KinematicMessagePassing(hidden_dim) for _ in range(num_layers)
        ])

        # Joint-level pooling
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temporal convolution (1D)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, width, 1),
            nn.ReLU(),
            *[nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.ReLU()) for _ in range(depth)],
        )

        # Downsampling
        ds = []
        for _ in range(down_t):
            ds.append(nn.Conv1d(width, width, stride_t * 2, stride_t, stride_t))
            ds.append(nn.ReLU())
        self.downsample = nn.Sequential(*ds)

        self.out_proj = nn.Conv1d(width, out_dim, 1)

    def forward(self, x, edge_index):
        """
        x: (B, T, J, in_dim) per-joint features per frame
        edge_index: (2, E) kinematic tree edges
        Returns: (B, out_dim, T_out)
        """
        B, T, J, C = x.shape

        # Flatten batch and time: (B*T, J, C)
        x_flat = x.reshape(B * T, J, C)

        # Per-frame spatial GNN
        x_flat = F.relu(self.joint_proj(x_flat))  # (BT, J, H)
        for layer in self.gnn_layers:
            x_flat = layer(x_flat, edge_index)

        # Pool across joints -> (BT, H)
        x_pooled = self.pool_proj(x_flat).mean(dim=1)

        # Reshape for temporal conv: (B, H, T)
        x_temporal = x_pooled.reshape(B, T, -1).permute(0, 2, 1)

        # Temporal convolution
        x_temporal = self.temporal(x_temporal)
        x_temporal = self.downsample(x_temporal)
        x_temporal = self.out_proj(x_temporal)  # (B, out_dim, T_out)

        return x_temporal


class KinematicMessagePassing(nn.Module):
    """
    Message passing along kinematic tree edges.
    For each parent->child: child receives message from parent.
    Residual connection + LayerNorm for stability.
    """
    def __init__(self, dim):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        """
        x: (N, D) features of all joints (N = BT * J)
        edge_index: (2, E) parent->child edges
        """
        parent_idx = edge_index[0]
        child_idx = edge_index[1]

        # Gather features along edges
        parent_feat = x[parent_idx]  # (E, D)
        child_feat = x[child_idx]    # (E, D)

        # Compute messages
        msg = self.msg_net(torch.cat([parent_feat, child_feat], dim=-1))

        # Scatter to children (sum aggregation)
        updates = torch.zeros_like(x)
        updates.scatter_add_(0, child_idx.unsqueeze(-1).expand(-1, x.shape[-1]), msg)

        # Residual + norm
        x_r = x + F.relu(updates)
        x_r = self.norm(x_r)

        return x_r


def build_kinematic_edges(kinematic_tree, bidirectional=True):
    """
    Build edge_index from kinematic tree.
    
    Args:
        kinematic_tree: List[List[int]] joint chains
        bidirectional: add child->parent edges too
    Returns:
        edge_index: (2, E) LongTensor
    """
    edges = []
    for chain in kinematic_tree:
        for j in range(1, len(chain)):
            edges.append([chain[j-1], chain[j]])
            if bidirectional:
                edges.append([chain[j], chain[j-1]])
    return torch.tensor(edges, dtype=torch.long).T