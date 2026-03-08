import torch
import torch.nn as nn
import torch.nn.functional as F


class UnknownTokenBank(nn.Module):
    """Unknown Token Bank: replaces single wildcard with K learnable prototypes.

    Each unknown region gets a region-specific token via top-m sparse softmax
    assignment over the bank, instead of sharing one "object" embedding.
    """

    def __init__(self, k, text_dim, static_embeddings, temperature=0.1, top_m=2):
        """
        Args:
            k: number of tokens in the bank
            text_dim: dimension of text embeddings (e.g. 1024 for RN50)
            static_embeddings: [K, text_dim] tensor of CLIP-encoded text prototypes
            temperature: softmax temperature for assignment
            top_m: number of top tokens to keep in sparse assignment
        """
        super().__init__()
        assert static_embeddings.shape == (k, text_dim)
        self.k = k
        self.text_dim = text_dim
        self.temperature = temperature
        self.top_m = top_m

        # Static text prototypes (frozen)
        self.register_buffer("static_tokens", static_embeddings)
        # Learnable offsets (init zeros, frozen during warmup)
        self.offsets = nn.Parameter(torch.zeros(k, text_dim))
        # Cache last assignment weights for balance loss computation
        self._last_weights = None
        # Warmup flag: when True, offsets are zeroed (but stay in graph for DDP)
        self.warmup = False

    def get_tokens(self):
        """Returns L2-normalized bank tokens: norm(static + offset). Shape [K, D]."""
        offsets = self.offsets * (0.0 if self.warmup else 1.0)
        return F.normalize(self.static_tokens + offsets, dim=-1)

    def assign(self, region_features):
        """Assign region features to bank tokens via top-m sparse softmax.

        Args:
            region_features: [N, D] L2-normalized region visual features

        Returns:
            assigned_tokens: [N, D] weighted combination of bank tokens
            weights: [N, K] sparse assignment weights
        """
        tokens = self.get_tokens()  # [K, D]
        # Cosine similarity (both already normalized)
        logits = region_features @ tokens.t() / self.temperature  # [N, K]

        # Top-m sparse softmax
        if self.top_m < self.k:
            topk_vals, topk_idx = logits.topk(self.top_m, dim=-1)  # [N, top_m]
            sparse_logits = torch.full_like(logits, float("-inf"))
            sparse_logits.scatter_(1, topk_idx, topk_vals)
            weights = F.softmax(sparse_logits, dim=-1)  # [N, K], sparse
        else:
            weights = F.softmax(logits, dim=-1)

        assigned_tokens = weights @ tokens  # [N, D]
        self._last_weights = weights.detach()  # cache for balance_loss (detached to avoid stale graph)
        return assigned_tokens, weights

    def diversity_loss(self):
        """Encourage bank tokens to be diverse (low pairwise cosine similarity)."""
        tokens = self.get_tokens()  # [K, D]
        gram = tokens @ tokens.t()  # [K, K]
        # Zero out diagonal, square off-diagonal entries
        mask = 1.0 - torch.eye(self.k, device=gram.device)
        loss = (gram ** 2 * mask).sum() / (self.k * (self.k - 1))
        return loss

    def balance_loss(self, weights):
        """Encourage balanced assignment by maximizing entropy of mean assignment.

        Args:
            weights: [N, K] assignment weights from assign()

        Returns:
            Negative entropy of mean assignment (minimize this to maximize entropy).
        """
        if weights is None or weights.numel() == 0:
            return torch.tensor(0.0, device=self.static_tokens.device)
        p_bar = weights.mean(dim=0)  # [K]
        p_bar = p_bar.clamp(min=1e-8)  # avoid log(0)
        entropy = -(p_bar * p_bar.log()).sum()
        return -entropy  # minimize → maximize entropy
