"""VCC perturbation dataset backed directly by compact context NPZ files.

The stock DNA/token-panel logic stays in ExpressionDataset.  This adapter only
replaces thousands of per-perturbation qref CSV files by the equivalent
expression/control arrays stored in one audited NPZ per context.
"""
from __future__ import annotations

import os
import pickle
import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from downstream_tasks.expression_prediction.expression_dataset_final_pert import (
    ExpressionDataset,
    logtransform,
)


class VCCQrefExpressionDataset(ExpressionDataset):
    def __init__(self, vcc_array_path: str, context_id: str, **kwargs):
        self.vcc_array_path = str(Path(vcc_array_path).resolve())
        self.context_id = str(context_id)
        with np.load(self.vcc_array_path, allow_pickle=False) as data:
            self.vcc_expression = data["expression"].astype(np.float32)
            self.vcc_control = data["control"].astype(np.float32)
            self.vcc_lfc = data["lfc_true"].astype(np.float32)
            self.vcc_sig = data["sig_mask"].astype(np.float32)
            self.vcc_targets = data["target_genes"].astype(str).tolist()
            self.vcc_genes = data["panel_genes"].astype(str).tolist()
            self.vcc_target_rows = data["target_gene_row"].astype(np.int64)

        expected = (len(self.vcc_targets), len(self.vcc_genes))
        for name, value in (
            ("expression", self.vcc_expression),
            ("lfc_true", self.vcc_lfc),
            ("sig_mask", self.vcc_sig),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape}; expected {expected}")
        if self.vcc_control.shape != (expected[1],):
            raise ValueError("control shape does not match panel_genes")
        if not np.isfinite(self.vcc_expression).all() or not np.isfinite(self.vcc_control).all():
            raise ValueError("expression/control contain NaN or Inf")
        if (self.vcc_expression < 0).any() or (self.vcc_control < 0).any():
            raise ValueError("qref expression/control must be non-negative")
        if not np.isin(self.vcc_sig, (0, 1)).all():
            raise ValueError("sig_mask must be binary")

        forward = pd.read_csv(kwargs["forward_intervals_path"], sep=None, engine="python")
        reverse = pd.read_csv(kwargs["reverse_intervals_path"], sep=None, engine="python")
        interval_genes = pd.concat([forward, reverse], ignore_index=True)
        symbol_to_id = dict(zip(interval_genes["gene_name"].astype(str), interval_genes["gene_id"].astype(str)))
        missing = [g for g in self.vcc_genes if g not in symbol_to_id]
        if missing:
            raise ValueError(f"{self.context_id}: {len(missing)} panel genes absent from intervals: {missing[:10]}")
        self._vcc_gene_ids = [symbol_to_id[g] for g in self.vcc_genes]
        self._vcc_target_ids = [symbol_to_id.get(g, g) for g in self.vcc_targets]
        self._vcc_gene_pos = {g: i for i, g in enumerate(self.vcc_genes)}

        cache_root = Path(os.environ.get("VCC_CACHE_DIR", Path(self.vcc_array_path).parent / ".cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(Path(self.vcc_array_path).read_bytes()).hexdigest()[:16]
        self._vcc_tpm_cache_path = str(
            cache_root / f"{self.context_id}.{digest}.tpm_lookup.pkl"
        )
        self._write_lookup_cache()
        super().__init__(**kwargs)

        if self.n_keys != len(self.vcc_targets):
            raise ValueError(
                f"n_keys must equal all {len(self.vcc_targets)} perturbations; got {self.n_keys}"
            )
        if self.track_keys != self.vcc_targets:
            raise ValueError("NPZ target order was not preserved")
        self.is_vcc_context_dataset = True

    def _write_lookup_cache(self):
        expected_keys = [f"{self.context_id}__non_targeting", *self.vcc_targets]
        if Path(self._vcc_tpm_cache_path).exists():
            try:
                with open(self._vcc_tpm_cache_path, "rb") as handle:
                    cached = pickle.load(handle)
                if list(cached) == expected_keys:
                    return
            except Exception:
                pass
        lookup = {
            expected_keys[0]: pd.DataFrame(
                self.vcc_control, index=self._vcc_gene_ids, columns=["expression"]
            )
        }
        for row, target in enumerate(self.vcc_targets):
            lookup[target] = pd.DataFrame(
                self.vcc_expression[row], index=self._vcc_gene_ids, columns=["expression"]
            )
        tmp = self._vcc_tpm_cache_path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            pickle.dump(lookup, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._vcc_tpm_cache_path)

    def get_tpm_hash_path(self):
        return self._vcc_tpm_cache_path

    def read_paths(self):
        dataset = self.context_id
        control = f"{dataset}__non_targeting"
        keys = [control, *self.vcc_targets]
        self.paths = {key: [{"+": None, "-": None}, self.vcc_array_path] for key in keys}
        self.dataset_description = dataset
        self.key_dataset = {key: dataset for key in keys}
        self.key_description = {key: dataset for key in keys}
        self.key_perturbation = {control: "non-targeting"}
        self.key_perturbation.update(dict(zip(self.vcc_targets, self._vcc_target_ids)))
        self.control_keys = {dataset: control}
        self.track_keys = list(self.vcc_targets)
        self.files_opened = False

    def build_queue(self):
        self.gene_tracks, self.queue = [], []
        available = set(self.vcc_genes)
        for idx, symbol in enumerate(self.genes["gene_name"].astype(str)):
            if symbol in available:
                self.queue.append((len(self.gene_tracks), 0))
                self.gene_tracks.append((idx, list(self.vcc_targets)))

    def _get_gene_key_order(self, gene_id: str, keys: List[str]) -> List[str]:
        if list(keys) != self.vcc_targets:
            raise ValueError(f"{self.context_id}/{gene_id}: perturbation order changed")
        return list(keys)

    def __getitem__(self, idx):
        gene_tracks_idx, _ = self.queue[idx]
        original_idx, _ = self.gene_tracks[gene_tracks_idx]
        symbol = str(self.genes.iloc[original_idx]["gene_name"])
        features = super().__getitem__(idx)
        if features["selected_keys"] != self.vcc_targets:
            raise ValueError(f"{self.context_id}/{symbol}: incomplete perturbation rectangle")
        g = self._vcc_gene_pos[symbol]
        control = float(self._transform_tpm(np.asarray([self.vcc_control[g]]))[0])
        features["control_value"] = torch.full(
            (len(self.vcc_targets),), control, dtype=torch.float32
        )
        features["lfc_true"] = torch.from_numpy(self.vcc_lfc[:, g].copy())
        features["sig_mask"] = torch.from_numpy(self.vcc_sig[:, g].copy())
        features["vcc_context_id"] = self.context_id
        features["vcc_gene_symbol"] = symbol
        features["vcc_target_genes"] = list(self.vcc_targets)
        return features
