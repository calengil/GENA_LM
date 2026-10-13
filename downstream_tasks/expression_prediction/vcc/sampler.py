"""Rectangular perturbation batch sampler.

Guarantees:
  * every batch is a complete rectangle: one context, ALL P perturbations of
    that context, a shared chunk of G response genes;
  * the batch is NEVER split across DDP ranks: the sampler accepts
    ``rank``/``num_replicas`` but requires ``num_replicas == 1`` on every rank,
    i.e. every rank receives the identical full batch (replicated data
    parallelism). DDP gradient averaging over identical batches is
    mathematically equivalent to single-rank training; the cost is duplicated
    FLOPs. Single-GPU training is recommended until a sharded-PDS variant
    (all-reduced P x P dot products) is implemented.
  * deterministic under a fixed seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple

from .contract import PerturbationDataAdapter

__all__ = ["BatchSpec", "RectangularPertBatchSampler"]


@dataclass(frozen=True)
class BatchSpec:
    """Everything the collate needs to build one rectangular batch."""

    context: str
    perturbations: Tuple[str, ...]
    genes: Tuple[str, ...]


class RectangularPertBatchSampler:
    """Yields :class:`BatchSpec` objects; one batch = one context chunk."""

    def __init__(
        self,
        adapter: PerturbationDataAdapter,
        batch_genes: int,
        seed: int = 45,
        shuffle: bool = True,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> None:
        if batch_genes < 1:
            raise ValueError("batch_genes must be >= 1")
        if num_replicas != 1:
            raise ValueError(
                "RectangularPertBatchSampler requires num_replicas == 1 on every rank: "
                "a P x G batch must not be split across DDP ranks because PDS needs "
                "all perturbation competitors in one process. Run single-GPU or use "
                "replicated batching (identical batches on all ranks)."
            )
        self.adapter = adapter
        self.batch_genes = batch_genes
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank  # accepted for Trainer compatibility; ignored by design

    def __len__(self) -> int:
        return sum(
            max(1, (len(self.adapter.response_genes(c)) + self.batch_genes - 1) // self.batch_genes)
            for c in self.adapter.contexts()
        )

    def __iter__(self) -> Iterator[BatchSpec]:
        rng = random.Random(self.seed)  # fresh generator: epoch-independent determinism
        contexts = list(self.adapter.contexts())
        if self.shuffle:
            rng.shuffle(contexts)
        for context in contexts:
            perts = list(self.adapter.perturbations(context))
            if len(perts) < 2:
                raise ValueError(
                    f"Context {context!r} has {len(perts)} perturbations; PDS needs >= 2"
                )
            genes = list(self.adapter.response_genes(context))
            if self.shuffle:
                rng.shuffle(genes)
            for start in range(0, len(genes), self.batch_genes):
                chunk = genes[start : start + self.batch_genes]
                yield BatchSpec(context=context, perturbations=tuple(perts), genes=tuple(chunk))
