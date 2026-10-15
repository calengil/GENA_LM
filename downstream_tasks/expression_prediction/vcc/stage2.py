"""Stage-2 members: SINGLE SHARED implementation for train loss and diagnostics.

Used by BOTH `vcc.composite.VCCCompositeLoss` (training) and
`vcc.qref_ladder` (gate / fixture diagnostics), so the gate always evaluates
exactly the formula that trains. There is no second implementation.

Key properties (fixed after the 2026-10 code audit):
  * the reference mask is GATED by the offline DE mask:
        w_ref = sig_mask * sigmoid((|lfc| - theta) / tau_s)
    so a Wilcoxon+BH sig_mask from prep_de_offline.py genuinely changes the
    training loss, and an all-zero sig_mask makes every Stage-2 member
    inactive (excluded from the composite);
  * each perturbation's own target-gene row is EXCLUDED per perturbation
    (competition rule), in every member;
  * soft Jaccard uses the Tanimoto product form
        L_jac = 1 - sum(w_hat*w_ref) / (sum(w_hat^2) + sum(w_ref^2) - sum(w_hat*w_ref))
    (for binary sets this is exactly 1 - |A&B|/|A|+|B|-|A&B| = 1 - |A&B|/|A|B|;
    at perfect predictions the loss is exactly 0 and the gradient is
    grad_i = -w_i * sum((w_hat-w_ref)^2) / D^2 -> 0 QUADRATICALLY, so float
    roundtrip noise cannot resurrect a gradient — unlike the min/max form,
    whose gradient at a near-tie is the one-sided subgradient of magnitude
    |dw_hat/dlfc|/den). Amplitude scaling w_hat = c*w_ref is stationary at
    c = 1, so the loss does not push amplitude inflation;
  * eps is unified at 1e-9 (the fixture LFC convention) everywhere.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

__all__ = [
    "pred_lfc",
    "soft_mask",
    "sign_agreement",
    "stage2_members_per_pert",
    "stage2_loss_values",
    "stage2_members",  # pooled single-perturbation wrapper (synthetic gate)
    "exact_members_fixed_theta",  # hard-mask stand-in (synthetic gate)
]

EPS = 1e-9  # unified with the fixture LFC convention


def pred_lfc(logits: torch.Tensor, control: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Differentiable map from log1p(TPM) space to predicted log2FC.

    NOTE (documented mismatch): the competition LFC is log2 of the ratio of
    arithmetic means of normalized counts; this per-row transform is the
    training-space surrogate. Closeness to the exact metric is what the
    monotonicity gate measures.
    """
    # Model outputs are unconstrained, while log1p(TPM) is non-negative.
    # Keep Stage-2 arithmetic in fp32 under mixed precision and prevent
    # log2 of a negative ratio during early training.
    logits = logits.float().clamp(min=0.0, max=20.0)
    control = control.float()
    if not torch.isfinite(control).all():
        raise ValueError("control contains NaN/Inf")
    if (control < 0).any():
        raise ValueError("control must be non-negative log1p(TPM)")
    d_hat = torch.expm1(logits)
    d0 = torch.expm1(control)
    # A 1e-9 pseudocount is adequate for values but gives d(log(x))/dx
    # near 1e9 at zero. Bound that derivative during early training.
    linear_floor = max(float(eps), 1e-4)
    d_hat = d_hat.clamp_min(linear_floor)
    d0 = d0.clamp_min(linear_floor)
    return torch.log2(d_hat / d0)


def soft_mask(lfc_abs: torch.Tensor, theta: float, tau_s: float) -> torch.Tensor:
    """w = sigmoid((|lfc| - theta) / tau_s). theta must be calibrated (> 0)."""
    if theta <= 0:
        raise ValueError("theta must be calibrated (> 0); theta=0 is not accepted")
    return torch.sigmoid((lfc_abs - theta) / tau_s)


def sign_agreement(lfc_hat: torch.Tensor, lfc: torch.Tensor, tau_d: float = 0.25) -> torch.Tensor:
    """a = 0.5 * (tanh(lfc_hat / tau_d) * tanh(lfc / tau_d) + 1) in [0, 1]."""
    s_hat = torch.tanh(lfc_hat / tau_d)
    s = torch.tanh(lfc / tau_d)
    return 0.5 * (s_hat * s + 1.0)


def calibrate_theta_from_sig(
    lfc: torch.Tensor,
    sig: torch.Tensor,
    sig_percentile: float = 0.3,
    theta_fallback: float = 0.585,
    min_sig_genes: int = 5,
) -> Optional[torch.Tensor]:
    """theta calibrated from the offline DE mask (plan §2 rule): the
    `sig_percentile` quantile of |lfc| among sig genes. This is how the
    (future Wilcoxon+BH) sig_mask participates in EVERY Stage-2 member:
    changing sig_mask changes theta changes the weights.

    Returns None when sig_mask is all-zero (members inactive); falls back to
    `theta_fallback` when fewer than `min_sig_genes` sig genes exist.
    """
    if not torch.is_nonzero(sig.sum()):
        return None
    sig_lfc = lfc[sig > 0.5].abs()
    sig_lfc = sig_lfc[torch.isfinite(sig_lfc) & (sig_lfc > 0)]
    if sig_lfc.numel() < min_sig_genes:
        return torch.tensor(theta_fallback, device=lfc.device, dtype=lfc.dtype)
    theta = torch.quantile(sig_lfc, sig_percentile)
    return theta.clamp_min(torch.finfo(lfc.dtype).eps)


def stage2_members_per_pert(
    lfc: torch.Tensor,
    lfc_hat: torch.Tensor,
    sig: torch.Tensor,
    target_gene_row: torch.Tensor,
    theta: Optional[float] = None,
    tau_s: float = 0.05,
    tau_d: float = 0.25,
    eps: float = EPS,
    sig_percentile: float = 0.3,
    theta_fallback: float = 0.585,
    min_sig_genes: int = 5,
) -> Dict[str, torch.Tensor]:
    """Per-perturbation Stage-2 member values.

    Args:
        lfc, lfc_hat, sig: (G*P,) row-major (gene, pert) vectors — true LFC
            (cell-eval2 convention, offline), predicted LFC (differentiable
            via `pred_lfc`), offline DE mask in {0,1}.
        target_gene_row: (P,) long, position of each perturbation's target
            gene within the response genes (-1 = not in the panel).
        theta: fixed threshold; if None, calibrated from sig_mask via
            `calibrate_theta_from_sig` (plan §2 rule) — this is how the
            offline DE mask (proxy or Wilcoxon+BH) participates in every
            member.

    Member semantics (cell-eval2-consistent):
        NMAE/FID/Reach are restricted to the DE set:
            w_ref = sig * sigmoid((|lfc| - theta)/tau_s);
        Jaccard is the Tanimoto product form over two soft sets of the SAME
        functional form (reference side NOT hard-gated, theta shared):
            L_jac = 1 - sum(w_hat*w_ref) /
                        (sum(w_hat^2) + sum(w_ref^2) - sum(w_hat*w_ref))
        so perfect predictions give exactly zero loss and a gradient that
        vanishes quadratically (grad_i = -w_i*sum((w_hat-w_ref)^2)/D^2).

    Returns:
        dict metric -> (P,) tensor of per-perturbation values; inactive
        perturbations (empty denominator, or all-zero sig) are NaN.
    """
    P = target_gene_row.numel()
    lfc_gp = lfc.view(-1, P)
    hat_gp = lfc_hat.view_as(lfc_gp)
    sig_gp = sig.view_as(lfc_gp)
    G = lfc_gp.shape[0]

    if theta is None:
        theta_t = calibrate_theta_from_sig(
            lfc, sig, sig_percentile, theta_fallback, min_sig_genes
        )
        if theta_t is None:
            # all-zero DE mask: every Stage-2 member inactive
            nan_col = torch.full((P,), float("nan"), device=lfc.device)
            return {m: nan_col.clone() for m in ("nmae", "fid", "reach", "jaccard")}
    else:
        theta_t = torch.tensor(float(theta), device=lfc.device)

    out: Dict[str, list] = {"nmae": [], "fid": [], "reach": [], "jaccard": []}
    for p in range(P):
        keep = torch.ones(G, dtype=torch.bool, device=lfc.device)
        t = int(target_gene_row[p])
        if 0 <= t < G:
            keep[t] = False  # per-perturbation competition exclusion
        lf, ht, sg = lfc_gp[keep, p], hat_gp[keep, p], sig_gp[keep, p]

        # NMAE/FID/Reach: reference weights restricted to the DE set
        w_ref = sg * soft_mask(lf.abs(), float(theta_t), tau_s)
        # Jaccard: soft sets of the same form on both sides (shared theta)
        w_ref_j = soft_mask(lf.abs(), float(theta_t), tau_s)
        w_hat_j = soft_mask(ht.abs(), float(theta_t), tau_s)
        a = sign_agreement(ht, lf, tau_d)

        den = w_ref.sum()
        if torch.is_nonzero(den):
            # Aggregate normalization avoids singular per-gene ratios when a
            # Wilcoxon-significant gene has exactly zero pseudobulk LFC.
            # It remains zero at perfect prediction and dimensionless.
            nmae_num = (w_ref * (ht - lf).abs()).sum()
            nmae_den = (w_ref * lf.abs()).sum()
            if nmae_den > eps:
                out["nmae"].append(nmae_num / nmae_den)
            else:
                out["nmae"].append(torch.nan)
            out["fid"].append((w_ref * (1.0 - a)).sum() / den)
        else:
            out["nmae"].append(torch.nan)
            out["fid"].append(torch.nan)

        den_r = (w_ref * ht.abs()).sum()
        if torch.is_nonzero(den_r):
            out["reach"].append((w_ref * ht.abs() * (1.0 - a)).sum() / den_r)
        else:
            out["reach"].append(torch.nan)

        # Tanimoto product-form soft Jaccard: zero loss at perfect, gradient
        # vanishes quadratically (see module docstring); reduces to the hard
        # set Jaccard for binary weights.
        num_j = (w_hat_j * w_ref_j).sum()
        den_j = (w_hat_j ** 2).sum() + (w_ref_j ** 2).sum() - num_j
        if torch.is_nonzero(den_j):
            out["jaccard"].append(1.0 - num_j / den_j)
        else:
            out["jaccard"].append(torch.nan)

    return {k: torch.stack([v if torch.is_tensor(v) else torch.tensor(v, device=lfc.device)
                            for v in vs]) for k, vs in out.items()}


def stage2_loss_values(
    lfc: torch.Tensor,
    lfc_hat: torch.Tensor,
    sig: torch.Tensor,
    target_gene_row: torch.Tensor,
    theta: Optional[float] = None,
    tau_s: float = 0.05,
    tau_d: float = 0.25,
    eps: float = EPS,
) -> Dict[str, torch.Tensor]:
    """Training-space member values: nanmean over perturbations (per-pert
    exclusion and DE-mask gating applied). Inactive members are absent.

    `theta=None` (default) calibrates the threshold from the offline DE mask
    via `calibrate_theta_from_sig` — this is how sig_mask participates in ALL
    four members, including Jaccard. A float theta fixes it."""
    per = stage2_members_per_pert(lfc, lfc_hat, sig, target_gene_row, theta, tau_s, tau_d, eps)
    out: Dict[str, torch.Tensor] = {}
    for k, v in per.items():
        if k == "nmae":
            continue
        finite = torch.isfinite(v)
        if finite.any():
            out[k] = v[finite].mean()

    # Global weighted NMAE over the complete rectangular batch. Averaging
    # per-perturbation ratios lets targets with tiny true effects dominate.
    P = target_gene_row.numel()
    lf_gp = lfc.view(-1, P)
    ht_gp = lfc_hat.view_as(lf_gp)
    sg_gp = sig.view_as(lf_gp)
    G = lf_gp.shape[0]
    theta_t = (calibrate_theta_from_sig(lfc, sig)
               if theta is None else torch.tensor(float(theta), device=lfc.device))
    if theta_t is not None:
        num = torch.zeros((), device=lfc.device, dtype=lfc.dtype)
        den = torch.zeros_like(num)
        for p in range(P):
            keep = torch.ones(G, dtype=torch.bool, device=lfc.device)
            t = int(target_gene_row[p])
            if 0 <= t < G:
                keep[t] = False
            lf, ht, sg = lf_gp[keep, p], ht_gp[keep, p], sg_gp[keep, p]
            w_ref = sg * soft_mask(lf.abs(), float(theta_t), tau_s)
            num = num + (w_ref * (ht - lf).abs()).sum()
            den = den + (w_ref * lf.abs()).sum()
        if den > eps:
            out["nmae"] = num / den
    return out


def stage2_members(
    lfc: torch.Tensor,
    lfc_hat: torch.Tensor,
    sig: torch.Tensor,
    theta: float,
    tau_s: float = 0.05,
    tau_d: float = 0.25,
    eps: float = EPS,
) -> Dict[str, torch.Tensor]:
    """Pooled single-perturbation wrapper (no exclusion; used by the synthetic
    gate tests where P=1 and there is no target row to exclude)."""
    row = torch.tensor([-1], dtype=torch.long)
    return {
        k: v[0] for k, v in stage2_members_per_pert(
            lfc.view(-1, 1), lfc_hat.view(-1, 1), sig.view(-1, 1), row,
            theta, tau_s, tau_d, eps,
        ).items()
    }


def exact_members_fixed_theta(
    lfc: torch.Tensor,
    lfc_hat: torch.Tensor,
    sig: torch.Tensor,
    theta: float,
    eps: float = EPS,
) -> Dict[str, torch.Tensor]:
    """Hard-mask exact-metric stand-in for the SYNTHETIC gate (single
    perturbation, no exclusion). Predicted significance uses the SAME fixed
    theta on both sides: true_sig = sig > 0.5, pred_sig = |lfc_hat| > theta.
    (An earlier draft derived the predicted threshold from
    min(|lfc| among true DE genes), which drops true DE genes at perfect
    predictions; fixed.)"""
    true_sig = sig > 0.5
    pred_sig = lfc_hat.abs() > theta
    out: Dict[str, torch.Tensor] = {}
    if true_sig.sum() >= 1:
        lf, ht = lfc[true_sig], lfc_hat[true_sig]
        out["nmae"] = torch.mean((ht - lf).abs() / (lf.abs() + eps))
        a_hard = ((ht > 0) == (lf > 0)).float()
        out["fid"] = 1.0 - a_hard.mean()
        den = ht.abs().sum()
        if torch.is_nonzero(den):
            out["reach"] = 1.0 - (ht.abs() * a_hard).sum() / den
    inter = (true_sig & pred_sig).float().sum()
    union = (true_sig | pred_sig).float().sum()
    if torch.is_nonzero(union) and torch.is_nonzero(true_sig.float().sum()):
        out["jaccard"] = 1.0 - inter / union
    return out
