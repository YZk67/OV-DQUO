import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPrototypeAggregator(nn.Module):
    """
    Learnable Text Prototype Aggregator with APR (Adaptive Prototype Regularization).

    Takes multi-prompt text embeddings [C, K_prompts, D] and produces
    K_proto prototypes per class [C, K_proto, D] via cross-attention.
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
        warmup_epochs=5,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.tau = max(float(tau), 1e-6)
        self.lambda_orth = float(lambda_orth)
        self.lambda_div = float(lambda_div)
        self.warmup_epochs = warmup_epochs

        # Cross-attention: prototype queries attend to multi-prompt keys/values
        self.key_proj = nn.Linear(text_dim, hidden_dim)
        self.value_proj = nn.Linear(text_dim, text_dim)
        self.prototype_queries = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # For APR warmup
        self.register_buffer("current_epoch", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_eye", torch.eye(num_prototypes), persistent=False)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.prototype_queries, std=0.02)

    def set_epoch(self, epoch):
        self.current_epoch.fill_(epoch)

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

        # Scaled dot-product attention with temperature
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
        """APR: orthogonality + diversity regularization with warmup."""
        epoch = self.current_epoch.item()
        warmup_scale = min(1.0, (epoch + 1) / max(self.warmup_epochs, 1))

        # Orthogonality: prototypes within same class should be orthogonal
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        sim_matrix = torch.bmm(proto_norm, proto_norm.transpose(1, 2))  # [C, K, K]
        eye = self._eye.to(sim_matrix.device)
        ortho_loss = (sim_matrix - eye).pow(2).mean()

        # Diversity: encourage spread of prototypes around class mean
        mean_proto = proto_norm.mean(dim=1, keepdim=True)  # [C, 1, D]
        div_loss = -(proto_norm - mean_proto).pow(2).sum(dim=-1).mean()

        apr_loss = warmup_scale * (self.lambda_orth * ortho_loss + self.lambda_div * div_loss)
        return apr_loss
