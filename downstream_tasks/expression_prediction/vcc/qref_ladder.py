"""Controlled prediction ladder + surrogate-to-exact gate for the qref fixture.

Ladder states (all valid log1p-space predictions, shape (G*P,)):
  perfect, small_noise, large_noise, sign_flipped, perturbation_shuffled,
  de_set_corrupted, constant_zero_delta.

Surrogate members: the SINGLE SHARED implementation
`vcc.stage2.stage2_loss_values` — identical to the training composite
(sig_mask gating + per-perturbation target-row exclusion).

Exact diagnostics: hard masks on BOTH sides with the SAME predeclared fixed
theta — true_sig = sig_mask > 0.5, pred_sig = |lfc_hat| > theta — per
perturbation with target-row exclusion, averaged over perturbations. These
are fixture_exact_metrics, NOT the official cell-eval2 vcc2026 score.

Gate criterion (FROZEN as of this version; to be validated on independent
prediction states before any production enablement):
  PASS iff Spearman rho >= 0.5 AND monotonicity violations <= 1 AND
  rho > mean(shuffle rho) + 0.2.
History: an earlier formulation used max(shuffle rho), which is degenerate
at n <= 7 states (reaches 1.0) and was replaced after an initial fixture
run; surrogate hyperparameters were never changed. The current criterion is
frozen and is validated on an independent state family (see
`build_validation_family`) before any production claim.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .contract import PertBatch
from .stage2 import EPS

# predeclared gate hyperparameters (from the synthetic gate / plan defaults;
# not tuned on the qref fixture)
GATE_THETA = 0.585        # log2(1.5)
GATE_TAU_SIGNIF = 0.05    # calibrated on the synthetic gate only
GATE_TAU_SIGN = 0.25
GATE_EPS = EPS            # unified 1e-9

LADDER_STATES = [
    "perfect", "small_noise", "large_noise", "sign_flipped",
    "perturbation_shuffled", "de_set_corrupted", "constant_zero_delta",
]


def build_ladder(batch: PertBatch, seed: int = 45) -> Dict[str, torch.Tensor]:
    """The 7 predeclared prediction states in batch row layout."""
    y, mu = batch.labels, batch.control_value
    G, P = batch.G, batch.P
    rng = torch.Generator().manual_seed(seed)

    small = torch.clamp(y + 0.05 * torch.randn(y.shape, generator=rng), min=0.0)
    large = torch.clamp(y + 0.50 * torch.randn(y.shape, generator=rng), min=0.0)

    perm = torch.tensor([1, 0, 3, 2]) if P == 4 else torch.randperm(P, generator=rng)
    shuffled = y.view(G, P)[:, perm].reshape(-1)

    sig = batch.sig_mask if batch.sig_mask is not None else torch.zeros_like(y)
    corrupted = torch.where(sig > 0.5, mu, torch.clamp(mu + 1.0, min=0.0))

    return {
        "perfect": y.clone(),
        "small_noise": small,
        "large_noise": large,
        "sign_flipped": torch.clamp(2.0 * mu - y, min=0.0),
        "perturbation_shuffled": shuffled,
        "de_set_corrupted": corrupted,
        "constant_zero_delta": mu.clone(),
    }


def build_validation_family(batch: PertBatch, seed: int = 123) -> Dict[str, torch.Tensor]:
    """Independent prediction-state family for gate validation: corruption
    types and noise seeds NOT used by the reporting ladder."""
    y, mu = batch.labels, batch.control_value
    G, P = batch.G, batch.P
    rng = torch.Generator().manual_seed(seed)
    sig = batch.sig_mask if batch.sig_mask is not None else torch.zeros_like(y)

    states: Dict[str, torch.Tensor] = {}
    # independent noise sweep (different seed and levels than the ladder)
    for i, s in enumerate((0.02, 0.10, 0.30, 0.80)):
        states[f"val_noise_{i}"] = torch.clamp(
            y + s * torch.randn(y.shape, generator=rng), min=0.0
        )
    # amplitude scaling of the true deltas (misplaced magnitude, right sign)
    for i, c in enumerate((0.5, 2.0)):
        states[f"val_delta_scaled_{i}"] = torch.clamp(mu + c * (y - mu), min=0.0)
    # partial shuffle: cyclic rotation of perturbations within each gene block
    states["val_cyclic_shift"] = y.view(G, P).roll(1, dims=1).reshape(-1)
    # DE-set corruption variant: inflate non-DE genes only, keep DE intact
    states["val_false_positives"] = torch.where(
        sig > 0.5, y, torch.clamp(mu + 1.5, min=0.0)
    )
    # DE-set corruption variant: shrink DE genes only
    states["val_false_negatives"] = torch.where(
        sig > 0.5, mu, y
    )
    return states


def lfc_hat_of(batch: PertBatch, pred: torch.Tensor, eps: float = GATE_EPS) -> torch.Tensor:
    """Predicted LFC via the shared differentiable transform."""
    from .stage2 import pred_lfc
    return pred_lfc(pred, batch.control_value, eps=eps)


def surrogate_per_pert(
    lfc: torch.Tensor, lfc_hat: torch.Tensor, sig: torch.Tensor, batch: PertBatch,
    theta: float | None = None, tau_s: float = GATE_TAU_SIGNIF, tau_d: float = GATE_TAU_SIGN,
    eps: float = GATE_EPS,
) -> Dict[str, float]:
    """Delegates to the SHARED per-perturbation implementation (identical to
    the training composite, INCLUDING the theta policy: None = calibrate from
    sig_mask); returns per-metric nanmean over perturbations."""
    from .stage2 import stage2_loss_values
    vals = stage2_loss_values(lfc, lfc_hat, sig, batch.target_gene_row,
                              theta=theta, tau_s=tau_s, tau_d=tau_d, eps=eps)
    return {k: float(v) for k, v in vals.items()}


def exact_per_pert(
    lfc: torch.Tensor, lfc_hat: torch.Tensor, sig: torch.Tensor, batch: PertBatch,
    theta: float = GATE_THETA, eps: float = GATE_EPS,
) -> Dict[str, float]:
    """Hard-mask exact-metric stand-ins with the SAME fixed theta on both
    sides: true_sig = sig > 0.5, pred_sig = |lfc_hat| > theta. Per-pert
    target-row exclusion; mean over perturbations.

    fixture_exact_metrics — NOT the official cell-eval2 vcc2026 score.
    """
    G, P = batch.G, batch.P
    lfc_gp, hat_gp, sig_gp = lfc.view(G, P), lfc_hat.view(G, P), sig.view(G, P)
    acc: Dict[str, List[float]] = {}
    for p in range(P):
        keep = torch.ones(G, dtype=torch.bool)
        t = int(batch.target_gene_row[p])
        if t >= 0:
            keep[t] = False
        lf, ht, sg = lfc_gp[keep, p], hat_gp[keep, p], sig_gp[keep, p]
        true_sig = sg > 0.5
        pred_sig = ht.abs() > theta  # SAME fixed theta on both sides
        vals: Dict[str, float] = {}
        if true_sig.sum() >= 1:
            lf_m, ht_m = lf[true_sig], ht[true_sig]
            vals["nmae"] = float(((ht_m - lf_m).abs() / (lf_m.abs() + eps)).mean())
            a_hard = ((ht_m > 0) == (lf_m > 0)).float()
            vals["fid"] = float(1 - a_hard.mean())
            den = ht_m.abs().sum()
            if torch.is_nonzero(den):
                vals["reach"] = float(1 - (ht_m.abs() * a_hard).sum() / den)
        inter = (true_sig & pred_sig).float().sum()
        union = (true_sig | pred_sig).float().sum()
        if torch.is_nonzero(union):
            vals["jaccard"] = float(1 - inter / union)
        for k, v in vals.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        return np.argsort(np.argsort(v)).astype(np.float64)
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _pairwise_concordance(x: np.ndarray, y: np.ndarray) -> float:
    """AUC-like ranking statistic: fraction of concordant pairs (ties -> 0.5)."""
    n = len(x)
    conc = []
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dy > 0:
                conc.append(1.0)
            elif dx * dy == 0:
                conc.append(0.5)
            else:
                conc.append(0.0)
    return float(np.mean(conc)) if conc else float("nan")


def _monotonicity_violations(x: np.ndarray, y: np.ndarray) -> int:
    """Violations of non-decreasing exact means across sorted surrogate order.

    NOTE: with 7 ladder states, decile binning is degenerate (1 sample per
    bin); the sorted sequence itself is reported instead.
    """
    order = np.argsort(x)
    s = y[order]
    return int(np.sum(np.diff(s) < -1e-12))


GATE_CRITERION = (
    "rho>=0.5 AND monotonicity_violations<=1 AND rho > shuffle_mean+0.2 "
    "(criterion FROZEN as of this version; validated on the independent "
    "state family before any production enablement)"
)


def _gate_for_family(
    batch: PertBatch, states: Dict[str, torch.Tensor], names: List[str],
    n_shuffle: int = 200, seed: int = 0,
) -> Tuple[List[dict], Dict[str, bool]]:
    lfc_true, sig = batch.lfc_true, batch.sig_mask
    assert lfc_true is not None and sig is not None

    sur_rows: Dict[str, List[float]] = {m: [] for m in ("nmae", "fid", "reach", "jaccard")}
    exa_rows: Dict[str, List[float]] = {m: [] for m in ("nmae", "fid", "reach", "jaccard")}
    for state in names:
        pred = states[state]
        lfc_hat = lfc_hat_of(batch, pred)
        sur = surrogate_per_pert(lfc_true, lfc_hat, sig, batch)
        exa = exact_per_pert(lfc_true, lfc_hat, sig, batch)
        for m in sur_rows:
            sur_rows[m].append(sur.get(m, float("nan")))
            exa_rows[m].append(exa.get(m, float("nan")))

    rng = np.random.default_rng(seed)
    rows, passed = [], {}
    for m in ("nmae", "fid", "reach", "jaccard"):
        sur = np.array(sur_rows[m], dtype=np.float64)
        exa = np.array(exa_rows[m], dtype=np.float64)
        keep = np.isfinite(sur) & np.isfinite(exa)  # undefined states excluded
        sur, exa = sur[keep], exa[keep]
        rho = _spearman(sur, exa)
        conc = _pairwise_concordance(sur, exa)
        viol = _monotonicity_violations(sur, exa)
        shuffle_rhos = [
            _spearman(rng.permutation(sur), exa) for _ in range(n_shuffle)
        ]
        shuffle_mean = float(np.mean(shuffle_rhos))
        ok = (rho >= 0.5) and (viol <= 1) and (rho > shuffle_mean + 0.2)
        passed[m] = ok
        rows.append({
            "metric": m, "n_states": int(keep.sum()),
            "spearman_rho": round(rho, 4),
            "pairwise_concordance": round(conc, 4),
            "monotonicity_violations": viol,
            "shuffle_control_mean_rho": round(shuffle_mean, 4),
            "gate": "PASS" if ok else "FAIL",
        })
    return rows, passed


def gate_table(
    batch: PertBatch, ladder: Dict[str, torch.Tensor], n_shuffle: int = 200,
    seed: int = 0,
) -> Tuple[List[dict], Dict[str, bool]]:
    """Per-metric gate rows over the ladder family + PASS/FAIL per metric."""
    return _gate_for_family(batch, ladder, LADDER_STATES, n_shuffle, seed)


def gate_table_validation(
    batch: PertBatch, states: Dict[str, torch.Tensor], n_shuffle: int = 200,
    seed: int = 0,
) -> Tuple[List[dict], Dict[str, bool]]:
    """Gate on the INDEPENDENT validation family (not the reporting ladder)."""
    return _gate_for_family(batch, states, list(states), n_shuffle, seed)
