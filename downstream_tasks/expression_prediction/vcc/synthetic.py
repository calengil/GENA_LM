"""Fully synthetic fixture: 1 context x 4 perturbations x 32 shared response genes.

Deterministic (seeded). No real data is read or implied. The fixture exercises
the full batch contract, including target genes that sit inside the response
panel (so PDS target-row exclusion is actually tested).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .contract import PerturbationDataAdapter

__all__ = ["SyntheticFixture", "build_default_fixture"]

# Special tokens for the panel vocabulary (mirrors the real tf-panel layout:
# [BOS] + one token per panel gene + [CLS]).
BOS_ID = 0
CLS_ID = 1
MASK_ID = 2
PANEL_BIN_OFFSET = 10  # binned expression tokens start here
N_PANEL_BINS = 50
DNA_VOCAB_OFFSET = 100  # DNA token ids start here
DNA_VOCAB_SIZE = 100


class SyntheticFixture(PerturbationDataAdapter):
    """In-memory synthetic adapter. Shapes: 1 context, P perturbations, G genes."""

    def __init__(
        self,
        n_perturbations: int = 4,
        n_genes: int = 32,
        seq_len: int = 64,
        seed: int = 45,
        context: str = "ctx_0",
    ) -> None:
        self.context = context
        self.P = n_perturbations
        self.G = n_genes
        self.L = seq_len
        self.rng = torch.Generator().manual_seed(seed)

        self.perts = [f"p_{i}" for i in range(n_perturbations)]
        self.genes = [f"g_{i:02d}" for i in range(n_genes)]
        # panel: [BOS] + one token per gene + [CLS]
        self.T = n_genes + 2
        self.panel_gene_pos = {g: i + 1 for i, g in enumerate(self.genes)}

        # DNA loci: one random token sequence per gene (constant across perts).
        self._dna = {
            g: torch.randint(DNA_VOCAB_OFFSET, DNA_VOCAB_OFFSET + DNA_VOCAB_SIZE, (seq_len,), generator=self.rng)
            for g in self.genes
        }

        # NTC control in log1p(TPM) space: sparse-ish expression profile.
        self._control = torch.rand(n_genes, generator=self.rng) * 4.0  # [0, 4]

        # Each perturbation targets one response gene (round-robin) so that
        # target-row exclusion is exercised for every perturbation.
        self._target_row = {p: i % n_genes for i, p in enumerate(self.perts)}

        # True latent deltas: a fixed shared direction + perturbation-specific
        # sparse component, so delta vectors are distinct but correlated.
        shared_dir = torch.randn(n_genes, generator=self.rng) * 0.1
        self._deltas = {}
        for p in self.perts:
            delta = shared_dir + torch.randn(n_genes, generator=self.rng) * 0.3
            delta[self._target_row[p]] -= 1.5  # knockdown of the target gene
            self._deltas[p] = delta

        # Post-pert labels in log1p space, clipped at 0 (TPM >= 0).
        self._labels = {
            p: torch.clamp(self._control + self._deltas[p], min=0.0) for p in self.perts
        }

        # Offline DE quantities in the cell-eval2 convention (synthetic stand-in:
        # here we reuse the latent deltas; a real adapter MUST take these from
        # the offline prep script, see DATA_FORMAT_SPEC.md).
        theta = 0.585  # log2(1.5)
        self._lfc = {p: self._deltas[p] for p in self.perts}
        self._sig = {p: (self._lfc[p].abs() > theta).float() for p in self.perts}

    # -- PerturbationDataAdapter -------------------------------------------

    def contexts(self) -> List[str]:
        return [self.context]

    def perturbations(self, context: str) -> List[str]:
        self._check_context(context)
        return list(self.perts)

    def response_genes(self, context: str) -> List[str]:
        self._check_context(context)
        return list(self.genes)

    def dna_tokens(self, context: str, gene: str) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_context(context)
        if gene not in self._dna:
            raise KeyError(f"Unknown gene {gene}")
        ids = self._dna[gene]
        return ids, torch.ones_like(ids)

    def panel_tokens(self, context: str, perturbation: str) -> torch.Tensor:
        self._check_context(context)
        if perturbation not in self.perts:
            raise KeyError(f"Unknown perturbation {perturbation}")
        bins = torch.randint(
            PANEL_BIN_OFFSET, PANEL_BIN_OFFSET + N_PANEL_BINS, (self.G,), generator=self.rng
        )
        panel = torch.cat(
            [torch.tensor([BOS_ID]), bins, torch.tensor([CLS_ID])]
        )
        # encode the perturbation by masking its target-gene position
        panel[self.panel_gene_pos[self.genes[self._target_row[perturbation]]]] = MASK_ID
        return panel

    def target_gene_row(self, context: str, perturbation: str) -> int:
        self._check_context(context)
        return self._target_row[perturbation]

    def label(self, context: str, perturbation: str, gene: str) -> float:
        self._check_context(context)
        return float(self._labels[perturbation][self.genes.index(gene)])

    def control_value(self, context: str, gene: str) -> float:
        self._check_context(context)
        return float(self._control[self.genes.index(gene)])

    def de_table(
        self, context: str, perturbation: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        self._check_context(context)
        return self._lfc[perturbation].clone(), self._sig[perturbation].clone()

    # -- helpers ------------------------------------------------------------

    def _check_context(self, context: str) -> None:
        if context != self.context:
            raise KeyError(f"Unknown context {context!r}; fixture has only {self.context!r}")


def build_default_fixture() -> SyntheticFixture:
    """The MVP fixture: 1 context x 4 perturbations x 32 shared genes."""
    return SyntheticFixture(n_perturbations=4, n_genes=32, seq_len=64, seed=45)
