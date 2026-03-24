"""
Inter-category Semantic Transfer (IST) module for OV-DQUO.

v1 (ISTModule): Directly modifies CLIP text embeddings → collapses.
v2 (ISTv2Module): Dual-path additive boost, GAT on text only → no visual awareness.
v3 (ISTv3Module): C²SRT-style, cross-attention injects visual context into GAT.
    - Text categories attend to roi visual features via cross-attention
    - GAT propagates image-aware category representations
    - Score: clip_score + gate * (visual_proj @ GAT_output.T)
    - Each image gets its own category prototypes (image-conditioned)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """Single GATv2 attention layer with sparse edge-list computation.

    For small graphs (N < 256), falls back to dense implementation.
    For large graphs (e.g. LVIS 1203 categories), uses sparse edge-list
    to avoid O(N²) memory: only computes attention on actual edges.
    """

    SPARSE_THRESHOLD = 256

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

        # Cached edge index for sparse mode
        self._edge_index = None
        self._edge_adj_device = None

    def _get_edge_index(self, adj):
        """Cache edge index from adjacency matrix (recompute if device changes)."""
        if self._edge_index is None or self._edge_adj_device != adj.device:
            src, dst = (adj > 0).nonzero(as_tuple=True)  # dst <- src edges
            self._edge_index = (src, dst)
            self._edge_adj_device = adj.device
        return self._edge_index

    def forward(self, h, adj):
        """
        h: [N, in_dim] node features
        adj: [N, N] adjacency matrix (adj[i][j] > 0 means j -> i edge)
        Returns: [N, out_dim]
        """
        N = h.size(0)
        if N < self.SPARSE_THRESHOLD:
            return self._forward_dense(h, adj, N)
        return self._forward_sparse(h, adj, N)

    def _forward_dense(self, h, adj, N):
        """Original dense implementation for small graphs (OV-COCO 65 cats)."""
        Wh = self.W(h).view(N, self.num_heads, self.head_dim)

        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1, -1)

        e = self.leaky_relu(Wh_i + Wh_j)
        e = (e * self.a.unsqueeze(0).unsqueeze(0)).sum(-1)

        mask = (adj > 0).unsqueeze(-1).expand(-1, -1, self.num_heads)
        e = e.masked_fill(~mask, float('-inf'))

        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)

        out = torch.einsum('ijh,jhd->ihd', alpha, Wh)
        out = out.reshape(N, -1)
        return out

    def _forward_sparse(self, h, adj, N):
        """Sparse edge-list implementation for large graphs (OV-LVIS 1203 cats).

        Only computes attention scores on existing edges → O(E) instead of O(N²).
        """
        H = self.num_heads
        D = self.head_dim

        Wh = self.W(h).view(N, H, D)  # [N, H, D]
        row, col = self._get_edge_index(adj)  # row=dst_node (i), col=src_node (j)
        E = row.size(0)

        # Compute attention only on edges: e_ij = a^T LeakyReLU(Wh_i + Wh_j)
        e_val = self.leaky_relu(Wh[row] + Wh[col])  # [E, H, D]
        e_val = (e_val * self.a.unsqueeze(0)).sum(-1)  # [E, H]

        # Sparse softmax: group by destination node (row)
        e_max = torch.full((N, H), float('-inf'), device=h.device)
        e_max.scatter_reduce_(0, row.unsqueeze(1).expand(-1, H), e_val, reduce='amax')
        e_val = e_val - e_max[row]  # numerical stability
        e_exp = e_val.exp()

        e_sum = torch.zeros(N, H, device=h.device)
        e_sum.scatter_add_(0, row.unsqueeze(1).expand(-1, H), e_exp)
        alpha = e_exp / (e_sum[row] + 1e-16)  # [E, H]
        alpha = self.dropout(alpha)

        # Weighted aggregation: out_i = sum_j alpha_ij * Wh_j
        msg = alpha.unsqueeze(-1) * Wh[col]  # [E, H, D]
        out = torch.zeros(N, H, D, device=h.device)
        out.scatter_add_(0, row.unsqueeze(1).unsqueeze(2).expand(-1, H, D), msg)

        out = out.reshape(N, -1)  # [N, out_dim]
        return out


class ISTv3Module(nn.Module):
    """
    C²SRT-style Inter-category Semantic Transfer with visual context.

    Architecture:
        1. text_proj(text_features) -> h_text  [C, hidden]
        2. Cross-attention: h_text attends to roi_features -> image-aware h  [B, C, hidden]
        3. GAT propagation on category graph -> refined h  [B, C, hidden]
        4. text_out(h) -> ist_text  [B, C, ist_dim]
        5. score = clip_score + gate * (visual_out(roi) @ ist_text.T)

    Key difference from v2: GAT receives visual context, so text prototypes
    are IMAGE-CONDITIONED — different images produce different prototypes.
    """

    def __init__(self, text_dim, hidden_dim=256, ist_dim=256,
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.ist_dim = ist_dim

        # Learnable gate: sigmoid(-2) ≈ 0.12
        self.gate = nn.Parameter(torch.full((1,), -2.0))

        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Visual context projection
        self.visual_ctx_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-attention: text categories attend to visual features
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATv2Layer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projections
        self.text_out = nn.Linear(hidden_dim, ist_dim)
        self.visual_out = nn.Linear(text_dim, ist_dim)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.text_proj, self.visual_ctx_proj, self.text_out, self.visual_out]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, text_features, adj, roi_features, clip_roi_features=None):
        """
        Full forward: cross-attention + GAT + dual-path scoring.

        Args:
            text_features: [C, text_dim] frozen CLIP text embeddings
            adj: [C, C] category adjacency matrix
            roi_features: [B, Q, text_dim] visual features for IST (layer4)
            clip_roi_features: [B, Q, text_dim] visual features for CLIP score (layer3).
                               If None, uses roi_features for both.

        Returns:
            scores: [B, Q, C] classification scores
        """
        if clip_roi_features is None:
            clip_roi_features = roi_features

        B, Q, _ = roi_features.shape
        C = text_features.size(0)

        # 1. Project text -> [C, hidden]
        h_text = self.text_proj(text_features)
        h = h_text.unsqueeze(0).expand(B, -1, -1)  # [B, C, hidden]

        # 2. Project visual context -> [B, Q, hidden]
        v_ctx = self.visual_ctx_proj(roi_features)

        # 3. Cross-attention: each category attends to visual features
        h_cross, _ = self.cross_attn(h, v_ctx, v_ctx)  # [B, C, hidden]
        h = self.cross_norm(h + h_cross)

        # 4. GAT propagation (per batch element; sparse for LVIS 1203 cats)
        for gat, ln in zip(self.gat_layers, self.gat_norms):
            h_list = []
            for b in range(B):
                h_list.append(gat(h[b], adj))
            h_new = torch.stack(h_list)  # [B, C, hidden]
            h = ln(h + F.elu(h_new))

        # 5. Project to IST space
        ist_text = self.text_out(h)  # [B, C, ist_dim]
        ist_text = F.normalize(ist_text, dim=-1)

        # 6. IST scoring in learned space
        ist_visual = self.visual_out(roi_features)  # [B, Q, ist_dim]
        ist_visual = F.normalize(ist_visual, dim=-1)
        ist_score = torch.bmm(ist_visual, ist_text.transpose(1, 2))  # [B, Q, C]

        # 7. Combine with CLIP score
        clip_score = clip_roi_features @ text_features.t()  # [B, Q, C]
        gate = self.gate.sigmoid()
        return clip_score + gate * ist_score, ist_score


# Keep old modules for backward compatibility
class ISTv2Module(nn.Module):
    """(Deprecated) v2 IST without visual context in GAT."""

    def __init__(self, text_dim, hidden_dim=512, ist_dim=256,
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.ist_dim = ist_dim
        self.gate = nn.Parameter(torch.zeros(1))
        self.text_input_proj = nn.Linear(text_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATv2Layer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.text_out_proj = nn.Linear(hidden_dim, ist_dim)
        self.visual_proj = nn.Linear(text_dim, ist_dim)
        self._init_weights()

    def _init_weights(self):
        for proj in [self.text_input_proj, self.text_out_proj, self.visual_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward_text(self, text_features, adj):
        h = self.text_input_proj(text_features)
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h = ln(h + F.elu(gat(h, adj)))
        ist_text = self.text_out_proj(h)
        return F.normalize(ist_text, dim=-1)

    def classify(self, roi_features, text_features, ist_text):
        clip_score = roi_features @ text_features.t()
        ist_visual = F.normalize(self.visual_proj(roi_features), dim=-1)
        ist_score = ist_visual @ ist_text.t()
        return clip_score + self.gate.sigmoid() * ist_score


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
            h = ln(h + F.elu(gat(h, adj)))
        delta = self.output_proj(h)
        parallel = (delta * text_features).sum(dim=-1, keepdim=True) * text_features
        delta_orth = delta - parallel
        delta_orth = F.normalize(delta_orth, dim=-1)
        refined = text_features + self.gate * delta_orth
        return F.normalize(refined, dim=-1)
