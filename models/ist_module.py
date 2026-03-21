"""
Inter-category Semantic Transfer (IST) module for OV-DQUO.

Uses GATv2 to propagate semantic knowledge from base categories to novel ones
via a pre-built category relationship graph. Refines CLIP text embeddings
before they are used for classification.

Reference: C²SRT (Category-Adaptive Cross-Modal Semantic Refinement and Transfer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """Single GATv2 attention layer with multi-head support."""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h: [N, in_dim] node features
        adj: [N, N] adjacency matrix (adj[i][j] > 0 means j -> i edge)
        Returns: [N, out_dim]
        """
        N = h.size(0)
        # Linear transform: [N, out_dim] -> [N, num_heads, head_dim]
        Wh = self.W(h).view(N, self.num_heads, self.head_dim)

        # GATv2: apply attention after concatenation (not before)
        # e_ij = a^T * LeakyReLU(W_a * [h_i || h_j])
        # For efficiency: compute for all pairs using broadcasting
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, heads, head_dim]
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1, -1)  # [N, N, heads, head_dim]

        e = self.leaky_relu(Wh_i + Wh_j)  # [N, N, heads, head_dim]
        e = (e * self.a.unsqueeze(0).unsqueeze(0)).sum(-1)  # [N, N, heads]

        # Mask: only attend to neighbors (adj > 0)
        mask = (adj > 0).unsqueeze(-1).expand(-1, -1, self.num_heads)  # [N, N, heads]
        e = e.masked_fill(~mask, float('-inf'))

        # Attention weights
        alpha = F.softmax(e, dim=1)  # softmax over source nodes (dim=1)
        alpha = self.dropout(alpha)

        # Weighted aggregation: [N, heads, head_dim]
        out = torch.einsum('ijh,jhd->ihd', alpha, Wh)
        out = out.reshape(N, -1)  # [N, out_dim]

        return out


class ISTModule(nn.Module):
    """
    Inter-category Semantic Transfer module.

    Takes raw CLIP text embeddings and refines them using a GATv2 graph
    that encodes category relationships. Knowledge flows from base
    categories to novel ones through the graph structure.

    Delta is L2-normalized so the learnable gate directly controls the
    rotation angle: gate=0.1 → arctan(0.1)≈5.7°.
    """

    def __init__(self, text_dim, hidden_dim=512, num_layers=2, num_heads=4,
                 dropout=0.1, gate_init=0.1):
        """
        Args:
            text_dim: dimension of CLIP text embeddings (1024 for RN50)
            hidden_dim: hidden dimension in GAT layers
            num_layers: number of GAT layers (default 2)
            num_heads: number of attention heads
            dropout: dropout rate
            gate_init: initial gate value, controls rotation angle via arctan(gate)
        """
        super().__init__()
        self.text_dim = text_dim
        self.num_layers = num_layers

        # Fixed gate: controls rotation angle via arctan(gate).
        # Not learnable — forces IST to always contribute, GAT only learns direction.
        self.gate = gate_init

        # Input projection
        self.input_proj = nn.Linear(text_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Layer(hidden_dim, hidden_dim, num_heads, dropout)
            )
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection back to text_dim
        self.output_proj = nn.Linear(hidden_dim, text_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, text_features, adj, novel_mask=None):
        """
        Args:
            text_features: [num_cats, text_dim] L2-normalized CLIP text embeddings
            adj: [num_cats, num_cats] adjacency matrix
            novel_mask: unused, kept for interface compatibility

        Returns:
            refined_features: [num_cats, text_dim] refined text embeddings (L2-normalized)
        """
        h = self.input_proj(text_features)  # [N, hidden_dim]

        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, adj)
            h_new = F.elu(h_new)
            h = ln(h + h_new)

        delta = self.output_proj(h)  # [N, text_dim]

        # Orthogonal projection: remove component parallel to original CLIP features
        parallel = (delta * text_features).sum(dim=-1, keepdim=True) * text_features
        delta_orth = delta - parallel

        # L2 normalize delta so gate directly controls rotation angle
        delta_orth = F.normalize(delta_orth, dim=-1)

        # rotation angle = arctan(gate), e.g. gate=0.1 → 5.7°
        refined = text_features + self.gate * delta_orth
        refined = F.normalize(refined, dim=-1)

        return refined
