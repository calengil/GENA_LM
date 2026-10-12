"""Tensor contract and adapter interface for the perturbation-aware VCC MVP.

This module is DATA-INDEPENDENT: no real (Orion/AWS) data is accessed, and no
assumptions are made about its file layout. Real datasets plug in exclusively
through the :class:`PerturbationDataAdapter` interface, backed by an external
preprocessing script (see DATA_FORMAT_SPEC.md).

Batch layout (one context per batch, rectangular P x G):
    rows are ordered row-major by (gene, perturbation):
        row i  <->  gene g = i // P, perturbation p = i % P
    so the first P rows belong to gene 0, the next P to gene 1, etc.
    This matches the ``dataset_flag (G, P)`` block structure used by the
    GENA-LM ``ExpressionCounts`` forward for DNA-encoder deduplication.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

__all__ = ["PerturbationDataAdapter", "PertBatch", "build_batch"]


class PerturbationDataAdapter(ABC):
    """Interface every real-data adapter must implement.

    All returned tensors are CPU ``torch.Tensor`` objects. The adapter is the
    ONLY place where dataset-specific storage details may appear.
    """

    @abstractmethod
    def contexts(self) -> List[str]:
        """All context ids, e.g. ['ctx_A']."""

    @abstractmethod
    def perturbations(self, context: str) -> List[str]:
        """Perturbation ids for a context; len >= 2 is required for PDS."""

    @abstractmethod
    def response_genes(self, context: str) -> List[str]:
        """Shared response-gene ids for a context (identical order for all perts)."""

    @abstractmethod
    def dna_tokens(self, context: str, gene: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """(input_ids (L,), attention_mask (L,)) long tensors for a gene locus."""

    @abstractmethod
    def panel_tokens(self, context: str, perturbation: str) -> torch.Tensor:
        """(T,) long tensor: NTC expression panel of the context with the
        perturbation's target-gene position replaced by the [MASK] token."""

    @abstractmethod
    def target_gene_row(self, context: str, perturbation: str) -> int:
        """Index of the perturbation's target gene within response_genes(context),
        or -1 if the target gene is not among the response genes."""

    @abstractmethod
    def label(self, context: str, perturbation: str, gene: str) -> float:
        """Post-perturbation log1p(TPM) of `gene` (training space)."""

    @abstractmethod
    def control_value(self, context: str, gene: str) -> float:
        """NTC-control log1p(TPM) of `gene` (model input, never a training label)."""

    def de_table(
        self, context: str, perturbation: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Optional offline DE quantities in the cell-eval2 convention:
        (lfc (G,) float32, sig_mask (G,) float32 in {0,1}).
        Stage-1 loss does not require them; diagnostics and Stage-2 do."""
        return None


@dataclass
class PertBatch:
    """Rectangular P x G batch for one context. See module docstring for layout."""

    context_id: str
    G: int
    P: int
    input_ids: torch.Tensor          # (G*P, L) long
    attention_mask: torch.Tensor     # (G*P, L) long
    dataset_flag: torch.Tensor       # (G, P) float32, all ones (DNA duplicated in group)
    tf_ids: torch.Tensor             # (G*P, T) long
    tf_attention_mask: torch.Tensor  # (G*P, T) long
    labels: torch.Tensor             # (G*P,) float32
    control_value: torch.Tensor      # (G*P,) float32
    pert_index: torch.Tensor         # (G*P,) long
    gene_index: torch.Tensor         # (G*P,) long
    target_gene_row: torch.Tensor    # (P,) long (-1 = target not among response genes)
    lfc_true: Optional[torch.Tensor] = None  # (G*P,) float32, offline cell-eval2 LFC
    sig_mask: Optional[torch.Tensor] = None  # (G*P,) float32, offline significance

    def validate(self) -> None:
        """Fail fast on contract violations."""
        n = self.G * self.P
        shapes = {
            "input_ids": (self.input_ids, (n, None)),
            "attention_mask": (self.attention_mask, (n, None)),
            "tf_ids": (self.tf_ids, (n, None)),
            "tf_attention_mask": (self.tf_attention_mask, (n, None)),
            "labels": (self.labels, (n,)),
            "control_value": (self.control_value, (n,)),
            "pert_index": (self.pert_index, (n,)),
            "gene_index": (self.gene_index, (n,)),
            "target_gene_row": (self.target_gene_row, (self.P,)),
            "dataset_flag": (self.dataset_flag, (self.G, self.P)),
        }
        for name, (t, expected) in shapes.items():
            second = expected[1] if len(expected) > 1 else None
            if t.shape[0] != expected[0] or (
                second is not None and t.dim() > 1 and t.shape[1] != second
            ):
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)}, expected first dim {expected[0]}"
                    + (f", second dim {second}" if second is not None else "")
                )
        if self.P < 2:
            raise ValueError(f"PDS requires P >= 2 perturbations per batch, got P={self.P}")
        if not torch.equal(self.pert_index, torch.arange(n) % self.P):
            raise ValueError("pert_index must follow the row-major (gene, pert) layout")
        if not torch.equal(self.gene_index, torch.arange(n) // self.P):
            raise ValueError("gene_index must follow the row-major (gene, pert) layout")
        if not torch.all(self.dataset_flag == 1.0):
            raise ValueError("dataset_flag must be all ones: DNA is duplicated within a gene block")


def build_batch(
    adapter: PerturbationDataAdapter,
    context: str,
    perturbations: List[str],
    genes: List[str],
) -> PertBatch:
    """Assemble a rectangular P x G batch from an adapter.

    Fails fast if the adapter cannot provide a complete rectangle (missing
    genes/panels/labels), so partial data can never silently enter training.
    """
    P, G = len(perturbations), len(genes)
    if P < 2:
        raise ValueError(f"A batch needs >= 2 perturbations for PDS, got {P}")
    if G < 1:
        raise ValueError("A batch needs >= 1 response gene")

    dna = [adapter.dna_tokens(context, g) for g in genes]
    L = dna[0][0].numel()
    for g, (ids, mask) in zip(genes, dna):
        if ids.numel() != L or mask.numel() != L:
            raise ValueError(f"Variable DNA length for gene {g}: {ids.numel()} vs {L}")

    panels = [adapter.panel_tokens(context, p) for p in perturbations]
    T = panels[0].numel()
    for p, panel in zip(perturbations, panels):
        if panel.numel() != T:
            raise ValueError(f"Variable panel length for pert {p}: {panel.numel()} vs {T}")

    rows = G * P
    input_ids = torch.stack([ids for ids, _ in dna]).repeat_interleave(P, dim=0)
    attention_mask = torch.stack([mask for _, mask in dna]).repeat_interleave(P, dim=0)
    tf_ids = torch.stack(panels).repeat(G, 1)
    tf_attention_mask = torch.ones_like(tf_ids)

    labels = torch.tensor(
        [adapter.label(context, p, g) for g in genes for p in perturbations],
        dtype=torch.float32,
    )
    control = torch.tensor(
        [adapter.control_value(context, g) for g in genes for p in perturbations],
        dtype=torch.float32,
    )

    # Offline DE quantities: per perturbation, sliced to the requested gene
    # chunk, then laid out row-major (row i <-> gene i//P, pert i%P).
    lfc_true = sig_mask = None
    all_genes = adapter.response_genes(context)
    gene_pos = [all_genes.index(g) for g in genes]
    de_per_pert = [adapter.de_table(context, p) for p in perturbations]
    if any(de is not None for de in de_per_pert):
        if any(de is None for de in de_per_pert):
            raise ValueError("de_table must be provided for ALL perturbations or none")
        for p, (lfc_all, sig_all) in zip(perturbations, de_per_pert):
            if lfc_all.numel() != len(all_genes) or sig_all.numel() != len(all_genes):
                raise ValueError(
                    f"de_table for pert {p} must cover the full response-gene set "
                    f"({len(all_genes)}), got {lfc_all.numel()}"
                )
        lfc_mat = torch.stack([
            de[0][gene_pos] for de in de_per_pert
        ])  # (P, G)
        sig_mat = torch.stack([de[1][gene_pos] for de in de_per_pert])  # (P, G)
        lfc_true = lfc_mat.T.reshape(-1)  # (G*P,) row-major
        sig_mask = sig_mat.T.reshape(-1)

    batch = PertBatch(
        context_id=context,
        G=G,
        P=P,
        input_ids=input_ids,
        attention_mask=attention_mask,
        dataset_flag=torch.ones(G, P, dtype=torch.float32),
        tf_ids=tf_ids,
        tf_attention_mask=tf_attention_mask,
        labels=labels,
        control_value=control,
        pert_index=torch.arange(rows) % P,
        gene_index=torch.arange(rows) // P,
        target_gene_row=torch.tensor(
            [adapter.target_gene_row(context, p) for p in perturbations], dtype=torch.long
        ),
        lfc_true=lfc_true,
        sig_mask=sig_mask,
    )
    batch.validate()
    return batch
