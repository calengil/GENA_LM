"""Tiny contract-compatible model for CPU smoke tests.

Stand-in for GENA-LM ``ExpressionCounts``: gene (DNA) encoder + panel encoder +
broadcast fusion + head, honoring the same tensor contract. Randomly
initialized, CPU-only, small enough for unit tests. It validates the BATCH
CONTRACT and LOSS GRADIENTS, not the real model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contract import PertBatch
from .synthetic import BOS_ID, CLS_ID, DNA_VOCAB_OFFSET, DNA_VOCAB_SIZE, N_PANEL_BINS, PANEL_BIN_OFFSET

__all__ = ["TinyExpressionModel"]

DNA_VOCAB = DNA_VOCAB_OFFSET + DNA_VOCAB_SIZE
PANEL_VOCAB = PANEL_BIN_OFFSET + N_PANEL_BINS + 3


class TinyExpressionModel(nn.Module):
    def __init__(
        self,
        hidden: int = 32,
        layers: int = 1,
        heads: int = 4,
        dna_vocab_size: int = DNA_VOCAB,
        panel_vocab_size: int = PANEL_VOCAB,
    ) -> None:
        super().__init__()
        self.gene_emb = nn.Embedding(dna_vocab_size, hidden)
        self.panel_emb = nn.Embedding(panel_vocab_size, hidden)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden,
            batch_first=True, dropout=0.0,
        )
        self.gene_encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.panel_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=2 * hidden,
                batch_first=True, dropout=0.0,
            ),
            num_layers=layers,
        )
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: PertBatch) -> torch.Tensor:
        # DNA encoder over (G*P, L) — a real run would deduplicate to G rows
        # via dataset_flag; the tiny model skips that optimization.
        x = self.gene_emb(batch.input_ids)
        x = self.gene_encoder(x, src_key_padding_mask=batch.attention_mask == 0)

        # Panel encoder over (G*P, T); its [CLS] embedding is broadcast onto
        # every DNA position (same fusion shape as the real model).
        t = self.panel_emb(batch.tf_ids)
        t = self.panel_encoder(t, src_key_padding_mask=batch.tf_attention_mask == 0)
        cls_pos = batch.tf_attention_mask.sum(dim=1) - 1
        panel_cls = t[torch.arange(t.size(0)), cls_pos]  # (G*P, H)

        h = self.ln(x + panel_cls[:, None, :])
        logits = self.head(h).squeeze(-1)                # (G*P, L)
        return logits[:, 0]                              # CLS position prediction
