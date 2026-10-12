"""Stage-1 VCC loss: MSE/Huber + PDS InfoNCE.

Stage-2 members (FID / Reach / NMAE / Jaccard) are deliberately NOT implemented
here: per the approved plan they may be added only after the monotonicity gate
(`test_surrogate_monotonicity`) shows each surrogate tracks its exact
cell-eval2 metric. The composite averages ACTIVE members only.

Conventions:
  * logits/labels live in log1p(TPM) space (the model's output space);
  * deltas are taken against the NTC control, which is a model INPUT
    (control_value), never a training label;
  * the perturbation's own target-gene row is excluded from PDS vectors
    (mirrors the competition rule that the perturbed gene's row is excluded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .contract import PertBatch

__all__ = ["VCCStage1Loss", "LossReport"]


@dataclass
class LossReport:
    """Per-member values; inactive members are None."""

    total: torch.Tensor
    huber: Optional[torch.Tensor] = None
    pds: Optional[torch.Tensor] = None

    def scalars(self) -> Dict[str, float]:
        out = {"total": float(self.total)}
        if self.huber is not None:
            out["huber"] = float(self.huber)
        if self.pds is not None:
            out["pds"] = float(self.pds)
        return out


class VCCStage1Loss(nn.Module):
    """L = mean of active members: Huber (or MSE) + PDS InfoNCE.

    Args:
        use_huber: True -> SmoothL1 (Huber) with ``huber_delta``; False -> MSE.
        huber_delta: Huber transition point in log1p(TPM) units.
        pds_temperature: InfoNCE temperature T for cosine similarities.
        exclude_target_rows: drop each perturbation's own target-gene row from
            its PDS vectors (competition-compatible; keep True).
        weight_huber, weight_pds: member weights (default 1/1; the composite
            normalizes by the number of ACTIVE members, so with both active
            and default weights this is the equal mean).
    """

    def __init__(
        self,
        use_huber: bool = True,
        huber_delta: float = 1.0,
        pds_temperature: float = 0.1,
        exclude_target_rows: bool = True,
        weight_huber: float = 1.0,
        weight_pds: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if pds_temperature <= 0:
            raise ValueError("pds_temperature must be > 0")
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.pds_temperature = pds_temperature
        self.exclude_target_rows = exclude_target_rows
        self.weight_huber = weight_huber
        self.weight_pds = weight_pds
        self.eps = eps

    def forward(self, logits: torch.Tensor, batch: PertBatch) -> LossReport:
        if logits.shape != batch.labels.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} != labels shape {tuple(batch.labels.shape)}"
            )
        report = LossReport(total=torch.zeros((), device=logits.device))
        active = 0

        # --- member 1: expression accuracy (competition: expr_mse family; the
        # control cancels in difference-based errors, so plain MSE/Huber on
        # log-space predictions is the faithful training surrogate).
        if self.use_huber:
            huber = nn.functional.smooth_l1_loss(
                logits, batch.labels, reduction="mean", beta=self.huber_delta
            )
        else:
            huber = nn.functional.mse_loss(logits, batch.labels, reduction="mean")
        report.huber = huber
        report.total = report.total + self.weight_huber * huber
        active += 1

        # --- member 2: PDS InfoNCE (competition: pds_cosine; surrogate:
        # softmax cross-entropy over perturbation competitors by cosine of
        # delta vectors).
        pds = self._pds_infonce(logits, batch)
        if pds is not None:
            report.pds = pds
            report.total = report.total + self.weight_pds * pds
            active += 1

        if active > 1:
            report.total = report.total / active
        return report

    def _pds_infonce(self, logits: torch.Tensor, batch: PertBatch) -> Optional[torch.Tensor]:
        P, G = batch.P, batch.G
        if P < 2:
            return None  # inactive: no competitors

        # (G, P) delta matrices against the per-gene control.
        control = batch.control_value.view(G, P)  # constant per gene across perts
        delta_pred = logits.view(G, P) - control
        delta_true = batch.labels.view(G, P) - control

        # Row masks: for ranking perturbation p, drop p's own target-gene row.
        row_masks = torch.ones(P, G, dtype=torch.bool, device=logits.device)
        if self.exclude_target_rows:
            for p in range(P):
                t = int(batch.target_gene_row[p])
                if t >= 0:
                    row_masks[p, t] = False

        cos = torch.zeros(P, P, device=logits.device)
        for p in range(P):
            mask = row_masks[p]
            if mask.sum() < 2:
                return None  # degenerate: < 2 genes left, cosine undefined
            v_pred = delta_pred[mask, p]                       # (G_p,)
            v_true = delta_true[mask, :]                       # (G_p, P)
            denom = v_pred.norm() * v_true.norm(dim=0) + self.eps
            cos[p, :] = (v_true.T @ v_pred) / denom            # cos(D̂_p, D_p')

        return nn.functional.cross_entropy(cos / self.pds_temperature, torch.arange(P, device=logits.device))
