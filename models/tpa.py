import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPrototypeAggregator(nn.Module):
    """
    Text Prototype Aggregator following LaMI-DETR.

    Takes multi-prompt text embeddings [C, K_prompts, D] and produces
    K_proto prototypes per class [C, K_proto, D] via cross-attention.

    Uses step-based warmup: TPA is frozen for the first `warmup_ratio` of
    total training iterations. Before warmup, output = mean of input prompts.
    """

    def __init__(
        self,
        text_dim,
        num_prototypes=4,
        hidden_dim=256,
        dropout=0.1,
        tau=0.07,
        lambda_orth=0.10,
        lambda_div=0.03,
        warmup_ratio=0.05,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)
        self.lambda_orth = float(lambda_orth)
        self.lambda_div = float(lambda_div)
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
        # key_proj: small init so initial attention ≈ uniform
        nn.init.normal_(self.key_proj.weight, std=1e-4)
        nn.init.zeros_(self.key_proj.bias)
        # value_proj: identity init so output ≈ input (preserve CLIP alignment)
        nn.init.eye_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.prototype_queries, std=1e-4)

    def set_total_steps(self, total_steps):
        """Set total training steps (epochs * iters_per_epoch)."""
        self.total_steps.fill_(total_steps)

    def step(self):
        """Increment step counter. Call once per training iteration."""
        self.current_step += 1

    @property
    def warmup_done(self):
        if self.total_steps.item() == 0:
            return True
        warmup_iters = int(self.total_steps.item() * self.warmup_ratio)
        return self.current_step.item() >= warmup_iters

    def forward(self, multi_prompt_embed, with_loss=True):
        """
        Args:
            multi_prompt_embed: [C, K_prompts, D] multi-prompt text embeddings (frozen)
        Returns:
            prototypes: [C, num_proto, D]
            apr_loss: scalar tensor or None
        """
        C, K, D = multi_prompt_embed.shape

        # Before warmup: return mean of input prompts expanded to K prototypes
        if self.training and not self.warmup_done:
            mean_embed = multi_prompt_embed.mean(dim=1, keepdim=True)  # [C, 1, D]
            mean_embed = F.normalize(mean_embed, p=2, dim=-1)
            prototypes = mean_embed.expand(C, self.num_prototypes, D)
            return prototypes, torch.tensor(0.0, device=multi_prompt_embed.device)

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
            apr_loss = self._compute_apr_loss(prototypes)

        return prototypes, apr_loss

    def _compute_apr_loss(self, prototypes):
        """APR: orthogonality + diversity regularization."""
        # Orthogonality: prototypes within same class should be orthogonal
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        sim_matrix = torch.bmm(proto_norm, proto_norm.transpose(1, 2))  # [C, K, K]
        eye = self._eye.to(sim_matrix.device)
        ortho_loss = (sim_matrix - eye).pow(2).mean()

        # Diversity: encourage spread of prototypes around class mean
        mean_proto = proto_norm.mean(dim=1, keepdim=True)  # [C, 1, D]
        div_loss = -(proto_norm - mean_proto).pow(2).sum(dim=-1).mean()

        apr_loss = self.lambda_orth * ortho_loss + self.lambda_div * div_loss
        return apr_loss
