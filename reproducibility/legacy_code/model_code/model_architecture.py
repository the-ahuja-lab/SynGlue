"""
SynGlue: Attention-based PROTAC property prediction model.

Architecture:
  Input:  (batch, 3, 4800)  — [warhead, linker, E3] GROVER embeddings
  Proj:   4800 → 512
  PosEmb: Learnable 3 × 512 positional embeddings (warhead=0, linker=1, E3=2)
  Transformer: 2-layer TransformerEncoder (8 heads, 2048 FF)
  AttnPool: Weighted fusion over the 3 components
  Heads:   DC50 (regression) + Dmax (regression)
"""

import torch
import torch.nn as nn


class ProtacAttentionModel(nn.Module):
    """Transformer with positional encoding for PROTAC component modeling."""

    def __init__(
        self,
        input_dim: int = 4800,
        proj_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        regressor_hidden: int = 256,
        regressor_dropout: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        self.pos_embed = nn.Embedding(3, proj_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(proj_dim, 1)
        self.head_dc50 = nn.Sequential(
            nn.Linear(proj_dim, regressor_hidden), nn.ReLU(),
            nn.Dropout(regressor_dropout), nn.Linear(regressor_hidden, 1),
        )
        self.head_dmax = nn.Sequential(
            nn.Linear(proj_dim, regressor_hidden), nn.ReLU(),
            nn.Dropout(regressor_dropout), nn.Linear(regressor_hidden, 1),
        )

    def forward(self, x):
        h = self.proj(x)
        positions = torch.arange(3, device=x.device)
        h = h + self.pos_embed(positions)
        h = self.transformer(h)
        attn_scores = self.attn_pool(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (h * attn_weights).sum(dim=1)
        return self.head_dc50(fused), self.head_dmax(fused), attn_weights
