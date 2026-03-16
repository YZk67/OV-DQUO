import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPrototypeAggregator(nn.Module):
    """
    Text Prototype Aggregator following LaMI-DETR.

    Takes multi-prompt text embeddings [C, K_prompts, D] and produces
    K_proto prototypes per class [C, K_proto, D] via cross-attention.

    Warmup only affects APR loss lambdas (cosine ramp 0→1).
    TPA itself is active from step 0.
    """

    def __init__(
        self,
        text_dim,
        num_prototypes=5,
        hidden_dim=256,
        dropout=0.05,
        tau=0.07,
        lambda_orth=0.10,
        lambda_div=0.03,
        warmup_ratio=0.05,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)
        self.lambda_orth_base = float(lambda_orth)
        self.lambda_div_base = float(lambda_div)
        self.warmup_ratio = warmup_ratio

        # Cross-attention: prototype queries attend to multi-prompt keys/values
        self.key_proj = nn.Linear(text_dim, hidden_dim)
        self.value_proj = nn.Linear(text_dim, text_dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # Step-based warmup tracking
        self.register_buffer("current_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("total_steps", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_eye", torch.eye(num_prototypes), persistent=False)

        self._init_parameters()

    def _init_parameters(self):
        # All projections: Xavier uniform (matching LaMI-DETR)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.prototype_queries.unsqueeze(0)).squeeze_(0)

    def set_total_steps(self, total_steps):
        """Set total training steps (epochs * iters_per_epoch)."""
        self.total_steps.fill_(total_steps)

    def step(self):
        """Increment step counter. Call once per training iteration."""
        self.current_step += 1

    @property
    def warmup_steps(self):
        total = self.total_steps.item()
        return int(total * self.warmup_ratio) if total > 0 else 0

    def _effective_lambdas(self):
        """Cosine ramp for APR lambdas: 0 → base over warmup steps."""
        ws = self.warmup_steps
        if ws == 0:
            return self.lambda_orth_base, self.lambda_div_base
        progress = min(1.0, (self.current_step.item() + 1) / float(ws))
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.lambda_orth_base * factor, self.lambda_div_base * factor

    def forward(self, multi_prompt_embed, with_loss=True):
        """
        Args:
            multi_prompt_embed: [C, K_prompts, D] multi-prompt text embeddings (frozen)
        Returns:
            prototypes: [C, num_proto, D]
            apr_loss: scalar tensor or None
        """
        C, K, D = multi_prompt_embed.shape

        keys = self.key_proj(multi_prompt_embed)       # [C, K, hidden_dim]
        values = self.value_proj(multi_prompt_embed)    # [C, K, D]

        queries = self.prototype_queries.unsqueeze(0).expand(C, -1, -1)  # [C, num_proto, hidden_dim]

        # Scaled dot-product attention
        attn_logits = torch.bmm(queries, keys.transpose(1, 2))  # [C, num_proto, K]
        attn_logits = attn_logits / (math.sqrt(keys.size(-1)) * self.tau)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        prototypes = torch.bmm(attn_weights, values)   # [C, num_proto, D]
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        apr_loss = None
        if with_loss and self.training:
            apr_loss = self._compute_apr_loss(prototypes, attn_logits)

        return prototypes, apr_loss

    def _compute_apr_loss(self, prototypes, attn_logits):
        """APR: orthogonality + diversity with cosine-ramped lambdas."""
        lambda_orth, lambda_div = self._effective_lambdas()

        # Orthogonality: off-diagonal of Gram matrix → 0
        P = F.normalize(prototypes, p=2, dim=-1)
        K = self.num_prototypes
        G = torch.bmm(P, P.transpose(1, 2))  # [C, K, K]
        eye = self._eye.to(G.device)
        off_mask = 1.0 - eye
        ortho_loss = ((G - eye).pow(2) * off_mask).sum(dim=(-2, -1)).mean() / (K * K - K)

        # Diversity: entropy of prototype usage (from attention logits)
        # attn_logits: [C, K, N_prompts]
        w = torch.softmax(attn_logits, dim=1)  # normalize over prototypes → [C, K, N]
        votes = w.sum(dim=-1)  # [C, K] how much each prototype is used
        p = votes / (votes.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=1) / math.log(K)  # normalized entropy [C]
        div_loss = entropy.mean()

        apr_loss = lambda_orth * ortho_loss + lambda_div * div_loss
        return apr_loss
