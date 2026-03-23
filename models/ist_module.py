"""
Inter-category Semantic Transfer (IST) module for OV-DQUO.

v1 (ISTModule): Directly modifies CLIP text embeddings via orthogonal projection + gate.
    Problem: fragile, limited improvement due to constrained modification space.

v2 (ISTv2Module): Dual-path architecture inspired by C²SRT.
    - Text path: GAT propagates category relationships in a learned space.
    - Visual path: Projects roi_features into the same space.
    - Classification: clip_score + gate * ist_score (additive boost).
    - CLIP text embeddings are NEVER modified — no collapse risk.
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


class ISTv2Module(nn.Module):
    """
    Dual-path Inter-category Semantic Transfer module.

    Architecture:
        Text path:   text_features -> input_proj -> GAT x L -> text_out  [C, ist_dim]
        Visual path: roi_features  -> visual_proj                        [B, Q, ist_dim]
        IST score:   visual_proj @ text_out.T                            [B, Q, C]
        Final:       clip_score + sigmoid(gate) * ist_score

    Key properties:
        - CLIP text embeddings are NEVER modified -> no training collapse
        - IST operates in its own learned space -> no CLIP alignment constraint
        - Gate initialized to 0 -> sigmoid(0)=0.5, IST active from the start
          but clip_score dominates early since IST weights are random
        - GAT can freely learn inter-category knowledge transfer
    """

    def __init__(self, text_dim, hidden_dim=512, ist_dim=256,
                 num_layers=2, num_heads=4, dropout=0.1):
        """
        Args:
            text_dim: dimension of CLIP text embeddings (1024 for RN50, 512 for ViT-B/16)
            hidden_dim: hidden dimension in GAT layers
            ist_dim: output dimension of IST space (for visual-text matching)
            num_layers: number of GAT layers
            num_heads: number of attention heads in GAT
            dropout: dropout rate
        """
        super().__init__()
        self.text_dim = text_dim
        self.ist_dim = ist_dim

        # Learnable gate: sigmoid(gate) controls IST contribution
        # Init to 0 -> sigmoid(0) = 0.5
        self.gate = nn.Parameter(torch.zeros(1))

        # Text path: project text embeddings and propagate via GAT
        self.text_input_proj = nn.Linear(text_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATv2Layer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.text_out_proj = nn.Linear(hidden_dim, ist_dim)

        # Visual path: project roi features to IST space
        self.visual_proj = nn.Linear(text_dim, ist_dim)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.text_input_proj, self.text_out_proj, self.visual_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward_text(self, text_features, adj):
        """Pre-compute GAT-refined text prototypes in IST space.

        Args:
            text_features: [C, text_dim] frozen CLIP text embeddings
            adj: [C, C] adjacency matrix

        Returns:
            ist_text: [C, ist_dim] GAT-refined text prototypes
        """
        h = self.text_input_proj(text_features)  # [C, hidden_dim]

        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, adj)
            h_new = F.elu(h_new)
            h = ln(h + h_new)  # residual + layer norm

        ist_text = self.text_out_proj(h)  # [C, ist_dim]
        ist_text = F.normalize(ist_text, dim=-1)
        return ist_text

    def classify(self, roi_features, text_features, ist_text):
        """Dual-path classification: CLIP score + IST score.

        Args:
            roi_features: [B, Q, text_dim] CLIP visual features (L2-normalized)
            text_features: [C, text_dim] frozen CLIP text embeddings (L2-normalized)
            ist_text: [C, ist_dim] from forward_text()

        Returns:
            scores: [B, Q, C] classification scores (not softmaxed)
        """
        # Path 1: original CLIP similarity (frozen, always correct)
        clip_score = roi_features @ text_features.t()  # [B, Q, C]

        # Path 2: IST similarity in learned space
        ist_visual = self.visual_proj(roi_features)  # [B, Q, ist_dim]
        ist_visual = F.normalize(ist_visual, dim=-1)
        ist_score = ist_visual @ ist_text.t()  # [B, Q, C]

        # Combine: gate controls IST contribution
        gate = self.gate.sigmoid()  # (0, 1)
        return clip_score + gate * ist_score


# Keep old ISTModule for backward compatibility with existing checkpoints
class ISTModule(nn.Module):
    """(Deprecated) v1 IST that directly modifies text embeddings."""

    def __init__(self, text_dim, hidden_dim=512, num_layers=2, num_heads=4,
                 dropout=0.1, gate_init=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.gate = gate_init
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Layer(hidden_dim, hidden_dim, num_heads, dropout)
            )
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, text_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, text_features, adj, novel_mask=None):
        h = self.input_proj(text_features)
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, adj)
            h_new = F.elu(h_new)
            h = ln(h + h_new)
        delta = self.output_proj(h)
        parallel = (delta * text_features).sum(dim=-1, keepdim=True) * text_features
        delta_orth = delta - parallel
        delta_orth = F.normalize(delta_orth, dim=-1)
        refined = text_features + self.gate * delta_orth
        refined = F.normalize(refined, dim=-1)
        return refined
