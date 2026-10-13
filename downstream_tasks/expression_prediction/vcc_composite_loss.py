"""Stable Hydra/model adapter for the audited VCC loss implementation."""
from dataclasses import dataclass
from typing import Optional

import torch

from downstream_tasks.expression_prediction.vcc.composite import (
    CompositeLossReport,
    VCCCompositeLoss,
    arm_a_loss,
    arm_b_loss,
)


@dataclass
class VCCBatch:
    """Minimal duck-typed batch consumed by ``vcc.composite``."""

    G: int
    P: int
    labels: torch.Tensor
    control_value: torch.Tensor
    target_gene_row: torch.Tensor
    lfc_true: Optional[torch.Tensor] = None
    sig_mask: Optional[torch.Tensor] = None


__all__ = [
    "VCCBatch",
    "VCCCompositeLoss",
    "CompositeLossReport",
    "arm_a_loss",
    "arm_b_loss",
]
