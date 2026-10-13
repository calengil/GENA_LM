"""FileDatasetAdapter: reads the DATA_FORMAT_SPEC v1.0 tree.

This is the adapter real preprocessing scripts target. Adding a new source
(e.g. Orion) means producing a spec-compliant tree — the model and loss never
change. Validation is fail-fast at load time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .contract import PerturbationDataAdapter

__all__ = ["FileDatasetAdapter"]


class FileDatasetAdapter(PerturbationDataAdapter):
    def __init__(self, prep_root: str | Path) -> None:
        self.root = Path(prep_root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format_version") != "1.0":
            raise ValueError(f"unsupported format_version: {self.manifest.get('format_version')}")
        self._contexts = list(self.manifest["contexts"])
        self._meta: dict[str, dict] = {}
        for c in self._contexts:
            meta = json.loads((self.root / "contexts" / c / "meta.json").read_text())
            if not (meta.get("theta", 0) > 0):
                raise ValueError(f"context {c}: theta must be calibrated (> 0), got {meta.get('theta')}")
            if len(meta["perturbations"]) < 2:
                raise ValueError(f"context {c}: PDS needs >= 2 perturbations")
            self._meta[c] = meta
            self._validate_context(c)

    def _validate_context(self, context: str) -> None:
        meta = self._meta[context]
        ctx_dir = self.root / "contexts" / context
        G = len(meta["response_genes"])
        panel = np.load(ctx_dir / "panel.ntc.npy")
        if panel.shape[0] != len(meta["panel_genes"]) + 2:
            raise ValueError(f"context {context}: panel length {panel.shape[0]} != n_panel_genes+2")
        lo, hi = meta["panel_special_tokens"]["bin_offset"], meta["panel_special_tokens"]["n_bins"]
        inner = panel[1:-1]
        if not ((inner >= lo).all() and (inner < lo + hi).all()):
            raise ValueError(f"context {context}: panel bins out of range [{lo}, {lo + hi})")
        ctrl = np.load(ctx_dir / "control.npz")["mu_log1p"]
        if ctrl.shape != (G,):
            raise ValueError(f"context {context}: control shape {ctrl.shape} != ({G},)")
        for p in meta["perturbations"]:
            d = np.load(ctx_dir / "perts" / f"{p}.npz", allow_pickle=False)
            for key in ("labels_log1p", "lfc", "sig_mask"):
                if d[key].shape != (G,):
                    raise ValueError(f"context {context} pert {p}: {key} shape {d[key].shape} != ({G},)")

    # -- PerturbationDataAdapter -------------------------------------------

    def contexts(self) -> List[str]:
        return list(self._contexts)

    def perturbations(self, context: str) -> List[str]:
        return list(self._meta[context]["perturbations"])

    def response_genes(self, context: str) -> List[str]:
        return list(self._meta[context]["response_genes"])

    def dna_tokens(self, context: str, gene: str) -> Tuple[torch.Tensor, torch.Tensor]:
        d = np.load(self.root / "contexts" / context / "genes" / f"{gene}.npz")
        return torch.from_numpy(d["input_ids"].astype(np.int64)), torch.from_numpy(
            d["attention_mask"].astype(np.int64)
        )

    def panel_tokens(self, context: str, perturbation: str) -> torch.Tensor:
        panel = torch.from_numpy(np.load(self.root / "contexts" / context / "panel.ntc.npy"))
        meta = self._meta[context]
        pos = meta["response_genes"].index(
            str(np.load(self.root / "contexts" / context / "perts" / f"{perturbation}.npz")["target_gene"])
        )
        panel = panel.clone()
        panel[pos + 1] = meta["panel_special_tokens"]["mask"]
        return panel

    def target_gene_row(self, context: str, perturbation: str) -> int:
        d = np.load(self.root / "contexts" / context / "perts" / f"{perturbation}.npz")
        return int(d["target_gene_row"])

    def label(self, context: str, perturbation: str, gene: str) -> float:
        d = np.load(self.root / "contexts" / context / "perts" / f"{perturbation}.npz")
        return float(d["labels_log1p"][self._meta[context]["response_genes"].index(gene)])

    def control_value(self, context: str, gene: str) -> float:
        ctrl = np.load(self.root / "contexts" / context / "control.npz")["mu_log1p"]
        return float(ctrl[self._meta[context]["response_genes"].index(gene)])

    def de_table(
        self, context: str, perturbation: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        d = np.load(self.root / "contexts" / context / "perts" / f"{perturbation}.npz")
        return torch.from_numpy(d["lfc"]), torch.from_numpy(d["sig_mask"])
