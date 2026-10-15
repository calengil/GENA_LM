"""Differentiable FID / Reach surrogates v2 approximating cell-eval2 raw metrics.

Official anchors (cell-eval2 v0.18.0, `cell_eval2/metrics/direction.py`):

* `de_direction_fidelity_yield_raw` = k / max(n_pred, n_conf) per target
  (`_fidelity_family`, `_safe_div(k, pl.max_horizontal(n_pred, n_conf))`), where
  the scored set is {g : p_adj_pred < alpha} ∩ {real LFC defined} with the
  target gene excluded; k counts scored genes whose predicted LFC sign equals
  the real LFC sign; n_conf = |{g : p_adj_real < alpha, LFC defined}| minus the
  target gene. Over-calling is charged through n_pred, under-calling through
  n_conf (issue #259 ruled intentional).

* `de_direction_reach(corrected=False)` = k* / n_conf per target, where the
  ranking pool is the ADJUDICATED universe (reference-significant genes only,
  target gene excluded), sorted by [predicted-significant first, predicted
  |LFC| descending]; depth k counts adjudicable pairs; purity P(k) =
  cum_matches / k; k* = the DEEPEST prefix with P(k) >= REACH_PURITY_FLOOR
  (0.9, calibrated constant, `direction.py:29`); n_conf == 0 -> NaN (inactive).

These are TRAIN SURROGATES, not the official metric: the official path
recomputes DE (Wilcoxon) on predicted CELLS, while the surrogates operate on
predicted LFC vectors with a differentiable predicted-DE probability.

Design notes (documented mismatches):
* soft-FID v2: w_pred is a per-gene predicted-DE probability. Variant A uses a
  global |lfc| threshold; variant B uses a variance-aware z-score (delta-method
  SE in log2 space). With theta_rule="budget" (or z_threshold=None) the
  per-perturbation threshold matches the reference call rate n_conf/G, which
  fixes the n_pred_soft ~ n_conf budget and charges over-/under-calling
  through DISPLACEMENT in k_soft rather than through the denominator.
* Reach v2: the ranking is DIFFERENTIABLE via pairwise soft ranks
  r_i = 1 + sum_{j != i} sigmoid((s_j - s_i)/tau_rank) over the pool
  (self-comparison excluded: the diagonal of the pairwise sigmoid matrix
  is zeroed, so no constant sigmoid(0)=0.5 shift)
  (O(n_conf^2) in the pool size, not G); purity P(k) uses soft top-k
  membership, and reach = max_k(k * gate_k)/n_conf reproduces the official
  DEEPEST qualifying prefix. A hard-argsort fallback (ranking="hard") is
  O(G log G) with no gradient through the ordering.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .stage2 import EPS, sign_agreement

__all__ = [
    "smooth_max",
    "w_pred_lfc",
    "w_pred_z",
    "soft_fid_v2_per_pert",
    "reach_v2_per_pert",
    "hard_fid_raw_per_pert",
    "hard_reach_raw_per_pert",
    "REACH_PURITY_FLOOR",
]

#: cell-eval2.metrics.direction.REACH_PURITY_FLOOR (calibrated, direction.py:29)
REACH_PURITY_FLOOR = 0.9

# predeclared surrogate hyperparameters (train artifacts only; NOT tuned on any
# test ladder — the ladder in the report is the evaluation, not the tuning set)
THETA_FALLBACK = 0.585      # same fallback as stage2.calibrate_theta_from_sig
TAU_WPRED = 0.05            # sigmoid temperature of w_pred (variant A)
Z_THRESHOLD = 2.0           # variant B default z threshold (None -> budget rule)
TAU_Z = 0.3                 # variant B sigmoid temperature (z units);
                            # Rev 3: 0.25 -> 0.3 to fit the <=10x-Huber
                            # gradient budget (10.02x -> ~8.4x measured)
TAU_RANK = 0.25             # soft top-k membership temperature (rank-score units)
TAU_MEMBER = 1.0            # soft top-k membership temperature (rank units)
TAU_GATE = 0.1             # soft purity-gate temperature (purity units)
SMOOTH_EPS = 1e-6           # smoothmax curvature: sqrt((a-b)^2 + SMOOTH_EPS)
BASE_DEPTHS = (25, 50, 100, 200, 400)
LN2 = 1.4426950408889634


def smooth_max(a: torch.Tensor, b: torch.Tensor, eps: float = SMOOTH_EPS) -> torch.Tensor:
    """Numerically stable differentiable max: (a+b+sqrt((a-b)^2+eps))/2.

    The sqrt form is bounded (unlike log-sum-exp with a large beta) and adds a
    known bias of at most sqrt(eps)/2 (~5e-4 at eps=1e-6) at a == b.
    """
    return 0.5 * (a + b + torch.sqrt((a - b) ** 2 + eps))


def calibrate_theta(lfc_true: torch.Tensor, sig: torch.Tensor,
                    sig_percentile: float = 0.3,
                    rule: str = "sig_p30") -> Optional[float]:
    """theta from the train artifacts (never from a ladder state).

    Rules:
      * "sig_p30" — 30th percentile of |lfc| among true sig genes (the stage2
        convention). Conservative: by construction it sits ABOVE ~30% of the
        true sig |lfc| values, so a sharp w_pred under-calls them and the
        soft-FID score has a floor of ~1 - q at perfect prediction.
      * "sig_min" — minimum |lfc| among true sig genes. Achieves near-zero
        perfect-prediction loss when the sig/non-sig |lfc| distributions are
        separated; may over-call on real data where they overlap.
    Returns None if the sig set is empty.
    """
    if not torch.is_nonzero(sig.sum()):
        return None
    sig_lfc = lfc_true[sig > 0.5].abs()
    sig_lfc = sig_lfc[torch.isfinite(sig_lfc) & (sig_lfc > 0)]
    if sig_lfc.numel() == 0:
        return THETA_FALLBACK
    if rule == "sig_p30":
        return float(torch.quantile(sig_lfc, sig_percentile))
    if rule == "sig_min":
        return float(sig_lfc.min())
    raise ValueError(f"unknown theta rule: {rule}")


def w_pred_lfc(lfc_hat: torch.Tensor, theta: float, tau: float = TAU_WPRED) -> torch.Tensor:
    """Variant A (LFC-only): w_pred = sigmoid((|lfc_hat| - theta) / tau)."""
    return torch.sigmoid((lfc_hat.abs() - theta) / tau)


def w_pred_z(lfc_hat: torch.Tensor, sigma_control: torch.Tensor, n_cells: torch.Tensor,
             mean_ref: Optional[torch.Tensor] = None,
             z_threshold: float = Z_THRESHOLD, tau_z: float = TAU_Z,
             eps: float = EPS) -> torch.Tensor:
    """Variant B (variance-aware): z_hat = |lfc_hat| / SE_log2;
    w_pred = sigmoid((z_hat - z_threshold) / tau_z).

    Units matter: lfc_hat is log2 fold-change, so the control SE must be in
    the SAME (log2) space. With sigma_control/mean_ref given in linear CPM
    space, the delta-method SE of log2 fold change is
        SE_log2 = sigma_control / (mean_ref * ln2 * sqrt(n)) + eps.
    Passing sigma_control already in log2 space (mean_ref=None) uses
        SE_log2 = sigma_control / sqrt(n) + eps.
    """
    n = n_cells.clamp(min=1).to(lfc_hat.dtype)
    if mean_ref is None:
        se = sigma_control / torch.sqrt(n) + eps
    else:
        se = sigma_control / (mean_ref * LN2 * torch.sqrt(n)) + eps
    z = lfc_hat.abs() / se
    return torch.sigmoid((z - z_threshold) / tau_z)


# --- Variant C: calibrated w_pred (Revision 3) ----------------------------
# Logistic map fitted on the K562 GWPS train artifact (6 targets, 12,282 kept
# genes; full-batch fit, LOO-verified per target). Features (standardized):
#   [|lfc_hat|, z_hat = |lfc_hat|/SE_log2, log1p(mean_ref), log1p(sigma)]
# with SE_log2 = sigma/(mean_ref*ln2*sqrt(n)) + eps (same delta-method SE as
# variant B). The per-perturbation BUDGET rule (quantile match of the call
# rate n_conf/G) is applied to the logistic SCORE, combining the feature map
# with variant B's call-rate calibration. Constants shipped in
# wpred_c_calibration.npz; provenance in FID_REACH_V2_REPORT.md Rev 3.
WPC_MEAN = (0.204183, 1.157274, 4.563579, 4.707065)
WPC_SCALE = (0.370727, 1.09263, 1.103071, 0.623867)
WPC_COEF = (-1.2918, 1.8443, 1.7138, -1.0104)
WPC_INTERCEPT = -1.9355
WPC_TAU = 0.1                 # sigmoid temperature on the budget-shifted score


def w_pred_calibrated(lfc_hat: torch.Tensor, sigma_control: torch.Tensor,
                      n_cells: torch.Tensor, mean_ref: torch.Tensor,
                      n_conf: torch.Tensor, G: int,
                      tau: float = WPC_TAU, eps: float = EPS) -> torch.Tensor:
    """Variant C: budget-calibrated logistic w_pred over kept genes.

    lfc_hat (n_kept,), sigma_control/mean_ref (n_kept,) linear CPM,
    n_cells scalar for this perturbation. Score is differentiable through
    |lfc_hat| and z_hat; the budget quantile is detached (constant shift).
    """
    n = n_cells.clamp(min=1).to(lfc_hat.dtype)
    se = sigma_control.to(lfc_hat.dtype) / (
        mean_ref.to(lfc_hat.dtype) * LN2 * torch.sqrt(n)) + eps
    f = torch.stack([lfc_hat.abs(), lfc_hat.abs() / se,
                     torch.log1p(mean_ref.to(lfc_hat.dtype)),
                     torch.log1p(sigma_control.to(lfc_hat.dtype))], dim=1)
    f = (f - torch.tensor(WPC_MEAN, dtype=lfc_hat.dtype, device=f.device)) / \
        torch.tensor(WPC_SCALE, dtype=lfc_hat.dtype, device=f.device)
    score = f @ torch.tensor(WPC_COEF, dtype=lfc_hat.dtype, device=f.device) \
        + WPC_INTERCEPT
    q = torch.quantile(score.detach(), 1.0 - n_conf.to(lfc_hat.dtype) / float(G))
    return torch.sigmoid((score - q) / tau)


def _per_pert_views(lfc_true, lfc_hat, sig, target_gene_row):
    P = target_gene_row.numel()
    lfc_gp, hat_gp, sig_gp = (lfc_true.view(-1, P), lfc_hat.view(-1, P), sig.view(-1, P))
    G = lfc_gp.shape[0]
    return lfc_gp, hat_gp, sig_gp, G, P


def _per_pert_vec(vec: Optional[torch.Tensor], p: int, keep: torch.Tensor,
                  G: int, P: int) -> Optional[torch.Tensor]:
    """Restrict a per-gene (G,), per-pair (G*P,) or per-pert (P,) vector to the
    kept (target-row-excluded) genes of perturbation p."""
    if vec is None:
        return None
    v = vec.reshape(-1)
    if v.numel() == G * P:
        return v.view(G, P)[keep, p]
    if v.numel() == G:
        return v[keep]
    if v.numel() == P:
        return v[p].expand(int(keep.sum()))
    raise ValueError(f"expected {G}, {P} or {G * P} elements, got {v.numel()}")


def _w_pred(ht: torch.Tensor, *, w_pred_variant: str, theta, theta_rule: str,
            n_conf, G: int, tau: float, sigma_control, n_cells, mean_ref,
            z_threshold, tau_z: float, eps: float) -> torch.Tensor:
    """Dispatch the predicted-DE probability for one perturbation's kept genes."""
    if w_pred_variant == "A":
        theta_p = theta
        if theta_p is None:  # budget rule: match the reference call rate
            theta_p = torch.quantile(ht.detach().abs(),
                                     1.0 - n_conf.to(ht.dtype) / float(G))
        return w_pred_lfc(ht, float(theta_p), tau)
    raise ValueError(f"unsupported in _w_pred: {w_pred_variant}")


def soft_fid_v2_per_pert(
    lfc_true: torch.Tensor, lfc_hat: torch.Tensor, sig: torch.Tensor,
    target_gene_row: torch.Tensor, *,
    theta: Optional[float] = None, theta_rule: str = "sig_p30",
    tau: float = TAU_WPRED,
    tau_d: float = 0.25, eps: float = EPS,
    w_pred_variant: str = "A",
    sigma_control: Optional[torch.Tensor] = None,
    n_cells: Optional[torch.Tensor] = None,
    mean_ref: Optional[torch.Tensor] = None,
    z_threshold: Optional[float] = Z_THRESHOLD, tau_z: float = TAU_Z,
) -> Dict[str, torch.Tensor]:
    """soft-FID v2 per perturbation: 1 - k_soft / smoothmax(n_pred_soft, n_conf).

    w_true = real Wilcoxon+BH sig_mask; w_pred = differentiable predicted-DE
    probability (variant A LFC-only or variant B variance-aware); a = soft sign
    agreement. Target-gene row excluded per perturbation. An all-zero true DE
    set makes the member INACTIVE (NaN), never NaN by 0/0.
    """
    lfc_gp, hat_gp, sig_gp, G, P = _per_pert_views(lfc_true, lfc_hat, sig, target_gene_row)
    if theta is None and theta_rule != "budget":
        theta = calibrate_theta(lfc_true, sig, rule=theta_rule)
        if theta is None:
            return {"soft_fid_v2": torch.full((P,), float("nan"), device=lfc_true.device)}
    out = []
    for p in range(P):
        keep = torch.ones(G, dtype=torch.bool, device=lfc_true.device)
        t = int(target_gene_row[p])
        if 0 <= t < G:
            keep[t] = False
        lf, ht, sg = lfc_gp[keep, p], hat_gp[keep, p], sig_gp[keep, p]
        n_conf = sg.sum()
        if not torch.is_nonzero(n_conf):
            out.append(torch.tensor(float("nan"), device=lfc_true.device))
            continue
        if w_pred_variant == "A":
            wp = _w_pred(ht, w_pred_variant="A", theta=theta, theta_rule=theta_rule,
                         n_conf=n_conf, G=G, tau=tau, sigma_control=None,
                         n_cells=None, mean_ref=None, z_threshold=z_threshold,
                         tau_z=tau_z, eps=eps)
        elif w_pred_variant == "B":
            assert sigma_control is not None and n_cells is not None, \
                "variant B requires sigma_control and n_cells"
            sig_g = _per_pert_vec(sigma_control, p, keep, G, P)
            n_g = _per_pert_vec(n_cells, p, keep, G, P)
            mr_g = _per_pert_vec(mean_ref, p, keep, G, P)
            se = (sig_g.to(ht.dtype) / (mr_g.to(ht.dtype) * LN2 * torch.sqrt(n_g.to(ht.dtype).clamp(min=1))) + eps
                  if mr_g is not None else
                  sig_g.to(ht.dtype) / torch.sqrt(n_g.to(ht.dtype).clamp(min=1)) + eps)
            z = ht.abs() / se
            z_thr = z_threshold
            if z_thr is None:  # budget rule: match the reference call rate
                z_thr = torch.quantile(z.detach(), 1.0 - n_conf.to(ht.dtype) / float(G))
            wp = torch.sigmoid((z - z_thr) / tau_z)
        elif w_pred_variant == "C":
            assert sigma_control is not None and n_cells is not None and mean_ref is not None, \
                "variant C requires sigma_control, n_cells and mean_ref"
            sig_g = _per_pert_vec(sigma_control, p, keep, G, P)
            n_g = _per_pert_vec(n_cells, p, keep, G, P)
            mr_g = _per_pert_vec(mean_ref, p, keep, G, P)
            wp = w_pred_calibrated(ht, sig_g, n_g, mr_g, n_conf, G)
        else:
            raise ValueError(f"unknown w_pred_variant: {w_pred_variant}")
        a = sign_agreement(ht, lf, tau_d)
        k_soft = (wp * sg * a).sum()
        n_pred_soft = wp.sum()
        score = k_soft / smooth_max(n_pred_soft, n_conf.to(k_soft.dtype))
        out.append(1.0 - score)
    return {"soft_fid_v2": torch.stack(out)}


def reach_v2_per_pert(
    lfc_true: torch.Tensor, lfc_hat: torch.Tensor, sig: torch.Tensor,
    target_gene_row: torch.Tensor, *,
    theta: Optional[float] = None, theta_rule: str = "sig_p30",
    tau: float = TAU_WPRED,
    tau_d: float = 0.25, tau_gate: float = TAU_GATE,
    tau_rank: float = TAU_RANK, tau_member: float = TAU_MEMBER,
    purity_floor: float = REACH_PURITY_FLOOR, eps: float = EPS,
    w_pred_variant: str = "A",
    sigma_control: Optional[torch.Tensor] = None,
    n_cells: Optional[torch.Tensor] = None,
    mean_ref: Optional[torch.Tensor] = None,
    z_threshold: Optional[float] = Z_THRESHOLD, tau_z: float = TAU_Z,
    ranking: str = "soft",
) -> Dict[str, torch.Tensor]:
    """Reach v2 per perturbation: differentiable deepest-pure-prefix depth.

    Adjudicated universe (true sig genes only, target row excluded), ranked by
    rank_score = w_pred * |lfc_hat| DESCENDING.

    ranking="soft" (default, DIFFERENTIABLE ranking): soft rank
        r_i = 1 + sum_{j != i} sigmoid((s_j - s_i)/tau_rank)
    (O(n_pool^2) time/memory in the POOL size = n_conf, not G); membership in
    the top-k prefix m_ik = sigmoid((k - r_i)/tau_member); purity
    P(k) = sum_i m_ik * a_i / sum_i m_ik with a = soft sign agreement; soft
    gate_k = sigmoid((P(k) - floor)/tau_gate); reach = max_k(k * gate_k)/n_conf
    — the max reproduces the official DEEPEST qualifying prefix (the sum of
    gates would over-count when purity dips and recovers). Gradients flow
    through BOTH the ranking (pairwise sigmoids) and the sign agreement.

    ranking="hard" (fallback for very large pools): hard argsort order,
    purity via cumsum(a); O(G log G) time, O(G) memory, no gradient through
    the ordering.
    """
    lfc_gp, hat_gp, sig_gp, G, P = _per_pert_views(lfc_true, lfc_hat, sig, target_gene_row)
    if theta is None and theta_rule != "budget":
        theta = calibrate_theta(lfc_true, sig, rule=theta_rule)
        if theta is None:
            return {"reach_v2": torch.full((P,), float("nan"), device=lfc_true.device)}
    if ranking not in ("soft", "hard"):
        raise ValueError(f"unknown ranking: {ranking}")
    out = []
    for p in range(P):
        keep = torch.ones(G, dtype=torch.bool, device=lfc_true.device)
        t = int(target_gene_row[p])
        if 0 <= t < G:
            keep[t] = False
        lf, ht, sg = lfc_gp[keep, p], hat_gp[keep, p], sig_gp[keep, p]
        pool = sg > 0.5
        n_conf = sg.sum()
        if not torch.is_nonzero(n_conf) or pool.sum() < 1:
            out.append(torch.tensor(float("nan"), device=lfc_true.device))
            continue
        lf_p, ht_p = lf[pool], ht[pool]
        if w_pred_variant == "A":
            wp = w_pred_lfc(ht_p, float(theta), tau)
        elif w_pred_variant == "B":
            assert sigma_control is not None and n_cells is not None, \
                "variant B requires sigma_control and n_cells"
            sig_g = _per_pert_vec(sigma_control, p, keep, G, P)[pool]
            n_g = _per_pert_vec(n_cells, p, keep, G, P)[pool]
            mr_g = _per_pert_vec(mean_ref, p, keep, G, P)
            mr_g = mr_g[pool] if mr_g is not None else None
            se = (sig_g.to(ht.dtype) / (mr_g.to(ht.dtype) * LN2 * torch.sqrt(n_g.to(ht.dtype).clamp(min=1))) + eps
                  if mr_g is not None else
                  sig_g.to(ht.dtype) / torch.sqrt(n_g.to(ht.dtype).clamp(min=1)) + eps)
            z = ht_p.abs() / se
            z_thr = z_threshold
            if z_thr is None:
                z_thr = torch.quantile(z.detach(), 1.0 - n_conf.to(ht.dtype) / float(G))
            wp = torch.sigmoid((z - z_thr) / tau_z)
        elif w_pred_variant == "C":
            assert sigma_control is not None and n_cells is not None and mean_ref is not None, \
                "variant C requires sigma_control, n_cells and mean_ref"
            sig_g = _per_pert_vec(sigma_control, p, keep, G, P)[pool]
            n_g = _per_pert_vec(n_cells, p, keep, G, P)[pool]
            mr_g = _per_pert_vec(mean_ref, p, keep, G, P)
            mr_g = mr_g[pool] if mr_g is not None else None
            wp = w_pred_calibrated(ht_p, sig_g, n_g, mr_g, n_conf, G)
        else:
            raise ValueError(f"unknown w_pred_variant: {w_pred_variant}")
        s = wp * ht_p.abs()                       # rank score, differentiable
        a = sign_agreement(ht_p, lf_p, tau_d)     # soft sign agreement
        n_pool = s.numel()
        depths = torch.arange(1, n_pool + 1, device=s.device, dtype=s.dtype)
        if ranking == "soft":
            # pairwise soft ranks: r_i = 1 + #{j != i : s_j > s_i} (soft).
            # The self term (j == i, sigmoid(0) = 0.5) is EXCLUDED by zeroing
            # the diagonal, matching the documented definition exactly.
            diff = (s.unsqueeze(0) - s.unsqueeze(1)) / tau_rank   # (n, n)
            eye = torch.eye(n_pool, device=s.device, dtype=s.dtype)
            r = 1.0 + (torch.sigmoid(diff) * (1.0 - eye)).sum(1)  # (n_pool,)
            # membership of gene i in the top-k prefix
            m = torch.sigmoid((depths.unsqueeze(1) - r.unsqueeze(0)) / tau_member)
            denom = m.sum(1)
            purity = torch.where(denom > 0, (m * a.unsqueeze(0)).sum(1) / denom,
                                 torch.full_like(denom, float("nan")))
        else:
            order = torch.argsort(-s.detach())
            purity = torch.cumsum(a[order], 0) / depths
        gate = torch.sigmoid((purity - purity_floor) / tau_gate)
        # DEEPEST qualifying prefix: max_k k*gate_k (the sum over depths would
        # count every qualifying depth, not the deepest one)
        reach = (depths * gate).max() / n_conf.to(gate.dtype)
        out.append(reach)
    return {"reach_v2": torch.stack(out)}


# ---------------------------------------------------------------------------
# hard exact raw metrics (official raw formulas, NO reference scaling; local
# reproduction — predicted significance via a fixed |lfc_hat| threshold, the
# documented stand-in for Wilcoxon-on-predicted-cells which needs cells)
# ---------------------------------------------------------------------------
def hard_fid_raw_per_pert(lfc_true, lfc_hat, sig, target_gene_row, theta: float,
                          eps: float = EPS) -> Dict[str, torch.Tensor]:
    """k / max(n_pred, n_conf); k = predicted-sig AND true-sig AND sign match.
    n_conf == 0 -> NaN (inactive)."""
    lfc_gp, hat_gp, sig_gp, G, P = _per_pert_views(lfc_true, lfc_hat.detach(), sig, target_gene_row)
    out = []
    for p in range(P):
        keep = torch.ones(G, dtype=torch.bool, device=lfc_true.device)
        t = int(target_gene_row[p])
        if 0 <= t < G:
            keep[t] = False
        lf, ht, sg = lfc_gp[keep, p], hat_gp[keep, p], sig_gp[keep, p]
        true_sig = sg > 0.5
        pred_sig = ht.abs() > theta
        n_conf = int(true_sig.sum())
        if n_conf == 0:
            out.append(torch.tensor(float("nan"), device=lfc_true.device))
            continue
        n_pred = int(pred_sig.sum())
        k = int((pred_sig & true_sig & (torch.sign(ht) == torch.sign(lf))).sum())
        out.append(torch.tensor(k / max(n_pred, n_conf) if max(n_pred, n_conf) > 0
                                else float("nan")))
    return {"hard_fid_raw": torch.stack(out)}


def hard_reach_raw_per_pert(lfc_true, lfc_hat, sig, target_gene_row,
                            purity_floor: float = REACH_PURITY_FLOOR) -> Dict[str, torch.Tensor]:
    """k* / n_conf over the adjudicated universe ranked by
    [predicted-sig first, |lfc_hat| desc]; k* = deepest prefix with purity >= 0.9."""
    lfc_gp, hat_gp, sig_gp, G, P = _per_pert_views(lfc_true, lfc_hat.detach(), sig, target_gene_row)
    theta = 0.0  # pred-sig flag only orders the sort; any finite split works
    out = []
    for p in range(P):
        keep = torch.ones(G, dtype=torch.bool, device=lfc_true.device)
        t = int(target_gene_row[p])
        if 0 <= t < G:
            keep[t] = False
        lf, ht, sg = lfc_gp[keep, p], hat_gp[keep, p], sig_gp[keep, p]
        pool = sg > 0.5
        n_conf = int(pool.sum())
        if n_conf == 0:
            out.append(torch.tensor(float("nan"), device=lfc_true.device))
            continue
        lf_p, ht_p = lf[pool], ht[pool]
        pred_sig = (ht_p.abs() > theta).long()
        order = torch.argsort(-(pred_sig * 1e6) - ht_p.abs())  # sig first, then |lfc| desc
        matches = (torch.sign(ht_p[order]) == torch.sign(lf_p[order])).float()
        cum_match = matches.cumsum(0)
        depths = torch.arange(1, n_conf + 1, device=lf.device, dtype=lf.dtype)
        purity = cum_match / depths
        qual = (purity >= purity_floor).nonzero(as_tuple=False).reshape(-1)
        k_star = int(depths[qual].max()) if qual.numel() > 0 else 0
        out.append(torch.tensor(k_star / n_conf))
    return {"hard_reach_raw": torch.stack(out)}
