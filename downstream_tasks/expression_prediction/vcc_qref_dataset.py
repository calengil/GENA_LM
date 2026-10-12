"""Bridge existing GENA-LM ``ExpressionDataset`` to consolidated VCC arrays.

The base dataset still performs DNA tokenization and TF-panel construction from
the exported qref profiles.  This subclass fixes perturbation order and adds
the offline tensors required by Arm A/B.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from downstream_tasks.expression_prediction.expression_dataset_final_pert import (
    ExpressionDataset,
)


class VCCQrefExpressionDataset(ExpressionDataset):
    def __init__(self, vcc_array_path: str, context_id: str, **kwargs):
        with np.load(vcc_array_path, allow_pickle=False) as data:
            self.vcc_expression = data["expression"].astype(np.float32)
            self.vcc_control = data["control"].astype(np.float32)
            self.vcc_lfc = data["lfc_true"].astype(np.float32)
            self.vcc_sig = data["sig_mask"].astype(np.float32)
            self.vcc_targets = data["target_genes"].astype(str).tolist()
            self.vcc_genes = data["panel_genes"].astype(str).tolist()
        self.context_id = str(context_id)
        self._vcc_gene_pos = {g: i for i, g in enumerate(self.vcc_genes)}
        super().__init__(**kwargs)
        if self.n_keys != len(self.vcc_targets):
            raise ValueError(
                f"n_keys must equal all {len(self.vcc_targets)} perturbations; "
                f"got n_keys={self.n_keys}"
            )
        if self.track_keys != self.vcc_targets:
            raise ValueError("targets mapping order differs from target_genes in VCC NPZ")
        self.is_vcc_context_dataset = True

    def build_queue(self):
        """Keep only response genes with a complete P-wide qref rectangle.

        Unlike the generic perturbation dataset, the targeted response row is
        retained here: the composite loss masks it where required.
        """
        self.gene_tracks = []
        self.queue = []
        common = None
        for key in self.track_keys:
            profile = self.tpm_lookup[key].iloc[:, 0]
            finite = np.isfinite(
                self._transform_tpm(profile.to_numpy(dtype=np.float32))
            )
            known = set(profile.index[finite])
            common = known if common is None else common.intersection(known)
        common = common or set()
        for idx, gene_id in enumerate(self.genes["gene_id"]):
            if gene_id in common:
                self.queue.append((len(self.gene_tracks), 0))
                self.gene_tracks.append((idx, list(self.track_keys)))

    def _get_gene_key_order(self, gene_id: str, keys: List[str]) -> List[str]:
        # Composite loss requires identical perturbation columns for every gene.
        if set(keys) != set(self.vcc_targets):
            missing = sorted(set(self.vcc_targets) - set(keys))
            raise ValueError(
                f"{self.context_id}/{gene_id}: incomplete perturbation rectangle; "
                f"missing {missing[:10]} ({len(missing)} total)"
            )
        return list(self.vcc_targets)

    def __getitem__(self, idx):
        gene_tracks_idx, _ = self.queue[idx]
        original_idx, _ = self.gene_tracks[gene_tracks_idx]
        symbol = str(self.genes.iloc[original_idx]["gene_name"])
        features = super().__getitem__(idx)
        if features["selected_keys"] != self.vcc_targets:
            raise ValueError(
                f"{self.context_id}/{symbol}: dataset returned a non-rectangular "
                "or reordered perturbation panel"
            )
        if symbol not in self._vcc_gene_pos:
            raise KeyError(f"interval gene {symbol!r} is absent from {self.context_id} panel")
        g = self._vcc_gene_pos[symbol]
        # Labels/control must use the same model-space transform (normally log1p).
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
