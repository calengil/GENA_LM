"""VCCCompositeLoss: full weighted composite (Arm A / Arm B configs).

Arm A = 1.0*Huber + 0.1*PDS
Arm B = 1.0*Huber + 0.1*PDS + w_nmae*NMAE + 0.1*FID + 0.1*Reach + w_jac*Jaccard
(start weights for NMAE/Jaccard set from the fixture gradient audit so that
no Stage-2 member dominates Huber by an order of magnitude; see the report).

Normalization modes (EXPLICIT, never silent):
  * normalize="none"        -> total = sum_k weight_k * member_k (the task's
                               Arm formulas; used for Arm A/B reporting);
  * normalize="active_mean" -> total = (sum_k weight_k * member_k over ACTIVE
                               members) / n_active (the existing
                               VCCStage1Loss behavior; separate config).

Stage-2 members come from the SINGLE SHARED implementation
`vcc.stage2.stage2_loss_values` (the same code the gate evaluates):
  * reference mask gated by the offline DE mask: w_ref = sig * sigmoid((|lfc|-theta)/tau_s);
  * per-perturbation target-gene-row exclusion in every member;
  * soft Jaccard in the Tanimoto product form 1 - sum(w_hat*w_ref)/
    (sum(w_hat^2)+sum(w_ref^2)-sum(w_hat*w_ref)) — zero loss at perfect
    predictions and a gradient that vanishes quadratically;
  * theta=None (default) calibrates the threshold from the offline DE mask
    (`calibrate_theta_from_sig`), so sig_mask participates in ALL four
    members including Jaccard; passing a float fixes the threshold;
  * eps unified at 1e-9.
An all-zero sig_mask makes every Stage-2 member inactive (excluded).

VCCStage1Loss is UNCHANGED and remains the Stage-1 path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from .contract import PertBatch
from .stage2 import EPS, pred_lfc, stage2_loss_values

__all__ = ["VCCCompositeLoss", "CompositeLossReport"]


@dataclass
class CompositeLossReport:
    total: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    normalize: str = "none"

    def scalars(self) -> Dict[str, float]:
        out = {"total": float(self.total)}
        for k, v in self.components.items():
            out[k] = float(v)
        return out


def pds_infonce(
    logits: torch.Tensor,
    batch: PertBatch,
    temperature: float = 0.1,
    exclude_target_rows: bool = True,
    eps: float = EPS,
) -> torch.Tensor | None:
    """PDS surrogate: softmax-CE over perturbation competitors by cosine of
    delta vectors; each perturbation's own target-gene row excluded."""
    P, G = batch.P, batch.G
    if P < 2:
        return None
    control = batch.control_value.view(G, P)
    delta_pred = logits.view(G, P) - control
    delta_true = batch.labels.view(G, P) - control

    row_masks = torch.ones(P, G, dtype=torch.bool, device=logits.device)
    if exclude_target_rows:
        for p in range(P):
            t = int(batch.target_gene_row[p])
            if t >= 0:
                row_masks[p, t] = False

    cos = torch.zeros(P, P, device=logits.device)
    for p in range(P):
        mask = row_masks[p]
        if mask.sum() < 2:
            return None
        v_pred = delta_pred[mask, p]
        v_true = delta_true[mask, :]
        denom = v_pred.norm() * v_true.norm(dim=0) + eps
        cos[p, :] = (v_true.T @ v_pred) / denom
    return nn.functional.cross_entropy(
        cos / temperature, torch.arange(P, device=logits.device)
    )


class VCCCompositeLoss(nn.Module):
    def __init__(
        self,
        # member switches
        use_huber: bool = True,
        use_pds: bool = True,
        use_nmae: bool = False,
        use_fid: bool = False,
        use_reach: bool = False,
        use_jaccard: bool = False,
        # member hyperparameters / weights
        huber_delta: float = 1.0,
        weight_huber: float = 1.0,
        pds_temperature: float = 0.1,
        weight_pds: float = 0.1,
        weight_nmae: float = 0.02,
        weight_fid: float = 0.1,
        # Revision 3 (user-authorized): reach weight 0.1 -> 1/30 so the v2
        # reach gradient fits the <=10x-Huber budget (25.9x/3 ~ 8.6x measured
        # on the real K562 batch); see FID_REACH_V2_REPORT.md Rev 3.
        weight_reach: float = 0.0333,
        weight_jaccard: float = 0.01,
        # stage-2 surrogate hyperparameters (predeclared; never tuned on the
        # reporting fixture). theta=None calibrates the threshold from the
        # offline DE mask (calibrate_theta_from_sig, fallback 0.585); a float
        # fixes it. Calibration depends only on (lfc_true, sig_mask), never
        # on predictions, so it is constant across a prediction family.
        theta: float | None = None,
        tau_signif: float = 0.05,
        tau_sign: float = 0.25,
        eps: float = EPS,
        # normalization mode; see module docstring
        normalize: str = "none",
        # FID/Reach surrogate variants (v2 approximates cell-eval2 raw metrics;
        # default v1 until v2 passes the acceptance ladder — see fid_reach_v2.py)
        fid_variant: str = "v1",          # "v1" | "soft_yield_v2"
        reach_variant: str = "v1",        # "v1" | "topk_prefix_v2"
        w_pred_variant: str = "A",   # v2 only: "A" LFC-only | "B" variance-aware
                                     # | "C" calibrated logistic (Rev 3)
        theta_rule: str = "sig_p30",      # v2 only, when theta is None
        # v2 variant-B inputs (per-gene control std in linear CPM space and
        # per-perturbation cell counts); required when w_pred_variant == "B"
        sigma_control: torch.Tensor | None = None,
        n_cells: torch.Tensor | None = None,
        mean_ref: torch.Tensor | None = None,
        reach_ranking: str = "soft",      # v2 reach: "soft" | "hard"
    ) -> None:
        super().__init__()
        if normalize not in ("none", "active_mean"):
            raise ValueError(f"unknown normalize mode: {normalize}")
        if theta is not None and theta <= 0:
            raise ValueError("fixed theta must be calibrated (> 0); pass None to calibrate from sig_mask")
        self.use = {
            "huber": use_huber, "pds": use_pds, "nmae": use_nmae,
            "fid": use_fid, "reach": use_reach, "jaccard": use_jaccard,
        }
        self.huber_delta = huber_delta
        self.weight = {
            "huber": weight_huber, "pds": weight_pds, "nmae": weight_nmae,
            "fid": weight_fid, "reach": weight_reach, "jaccard": weight_jaccard,
        }
        self.pds_temperature = pds_temperature
        self.theta = theta
        self.tau_signif = tau_signif
        self.tau_sign = tau_sign
        self.eps = eps
        self.normalize = normalize
        if fid_variant not in ("v1", "soft_yield_v2"):
            raise ValueError(f"unknown fid_variant: {fid_variant}")
        if reach_variant not in ("v1", "topk_prefix_v2"):
            raise ValueError(f"unknown reach_variant: {reach_variant}")
        if w_pred_variant not in ("A", "B", "C"):
            raise ValueError(f"unknown w_pred_variant: {w_pred_variant}")
        if theta_rule not in ("sig_p30", "sig_min", "budget"):
            raise ValueError(f"unknown theta_rule: {theta_rule}")
        if reach_ranking not in ("soft", "hard"):
            raise ValueError(f"unknown reach_ranking: {reach_ranking}")
        self.fid_variant = fid_variant
        self.reach_variant = reach_variant
        self.w_pred_variant = w_pred_variant
        self.theta_rule = theta_rule
        self.sigma_control = sigma_control
        self.n_cells = n_cells
        self.mean_ref = mean_ref
        self.reach_ranking = reach_ranking
        if w_pred_variant in ("B", "C") and (sigma_control is None or n_cells is None):
            raise ValueError(f"w_pred_variant={w_pred_variant!r} requires sigma_control and n_cells")
        if w_pred_variant == "C" and mean_ref is None:
            raise ValueError("w_pred_variant='C' requires mean_ref")

    def forward(self, logits: torch.Tensor, batch: PertBatch) -> CompositeLossReport:
        if logits.shape != batch.labels.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} != labels shape {tuple(batch.labels.shape)}"
            )
        comps: Dict[str, torch.Tensor] = {}

        if self.use["huber"]:
            comps["huber"] = nn.functional.smooth_l1_loss(
                logits, batch.labels, reduction="mean", beta=self.huber_delta
            )
        if self.use["pds"]:
            pds = pds_infonce(logits, batch, self.pds_temperature)
            if pds is not None:
                comps["pds"] = pds
        if any(self.use[m] for m in ("nmae", "fid", "reach", "jaccard")):
            if batch.lfc_true is None or batch.sig_mask is None:
                raise ValueError(
                    "Stage-2 members require batch.lfc_true/batch.sig_mask "
                    "(offline DE table in the cell-eval2 convention)"
                )
            lfc_hat = pred_lfc(logits, batch.control_value, eps=self.eps)
            # SINGLE SHARED implementation (same as the gate diagnostics):
            # sig_mask gating + per-perturbation target-row exclusion inside.
            members = stage2_loss_values(
                batch.lfc_true, lfc_hat, batch.sig_mask, batch.target_gene_row,
                theta=self.theta, tau_s=self.tau_signif, tau_d=self.tau_sign,
                eps=self.eps,
            )
            for m in ("nmae", "fid", "reach", "jaccard"):
                if self.use[m] and m in members:
                    comps[m] = members[m]
            # v2 surrogates replace the v1 fid/reach members when configured
            if self.fid_variant == "soft_yield_v2" and self.use["fid"]:
                from .fid_reach_v2 import soft_fid_v2_per_pert
                v2 = soft_fid_v2_per_pert(
                    batch.lfc_true, lfc_hat, batch.sig_mask, batch.target_gene_row,
                    theta=self.theta, theta_rule=self.theta_rule,
                    tau_d=self.tau_sign, eps=self.eps,
                    w_pred_variant=self.w_pred_variant,
                    sigma_control=self.sigma_control, n_cells=self.n_cells,
                    mean_ref=self.mean_ref,
                )
                if "soft_fid_v2" in v2:
                    vals = v2["soft_fid_v2"]
                    finite = torch.isfinite(vals)
                    if finite.any():
                        comps["fid"] = vals[finite].mean()
            if self.reach_variant == "topk_prefix_v2" and self.use["reach"]:
                from .fid_reach_v2 import reach_v2_per_pert
                v2 = reach_v2_per_pert(
                    batch.lfc_true, lfc_hat, batch.sig_mask, batch.target_gene_row,
                    theta=self.theta, theta_rule=self.theta_rule,
                    tau_d=self.tau_sign, eps=self.eps,
                    w_pred_variant=self.w_pred_variant,
                    sigma_control=self.sigma_control, n_cells=self.n_cells,
                    mean_ref=self.mean_ref, ranking=self.reach_ranking,
                )
                if "reach_v2" in v2:
                    vals = v2["reach_v2"]
                    finite = torch.isfinite(vals)
                    if finite.any():
                        # reach_v2_per_pert returns a SCORE (k*/n_conf, higher
                        # = better); the composite minimizes LOSSES, so convert
                        # (same convention as the v1 reach member, a loss).
                        comps["reach"] = 1.0 - vals[finite].mean()

        total = torch.zeros((), device=logits.device)
        n_active = 0
        for name, value in comps.items():
            total = total + self.weight[name] * value
            n_active += 1
        if self.normalize == "active_mean" and n_active > 0:
            total = total / n_active
        return CompositeLossReport(
            total=total, components=comps,
            weights={k: self.weight[k] for k in comps}, normalize=self.normalize,
        )


def arm_a_loss() -> VCCCompositeLoss:
    """Arm A: Huber + PDS, weighted sum (task config)."""
    return VCCCompositeLoss(
        use_huber=True, use_pds=True, normalize="none",
        weight_huber=1.0, weight_pds=0.1,
    )


def arm_b_loss() -> VCCCompositeLoss:
    """Arm B: full composite, weighted sum (task config with audited start
    weights for NMAE/Jaccard)."""
    return VCCCompositeLoss(
        use_huber=True, use_pds=True, use_nmae=True, use_fid=True,
        use_reach=True, use_jaccard=True, normalize="none",
        weight_huber=1.0, weight_pds=0.1, weight_nmae=0.02,
        weight_fid=0.1, weight_reach=0.0333, weight_jaccard=0.01,
    )
