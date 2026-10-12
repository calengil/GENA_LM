"""Perturbation-aware VCC MVP: batch contract, sampler, Stage-1 loss.

Data-independent by design: real datasets connect only through
``PerturbationDataAdapter`` implementations backed by an external
preprocessing script (see DATA_FORMAT_SPEC.md).
"""
from .contract import PertBatch, PerturbationDataAdapter, build_batch
from .composite import VCCCompositeLoss, arm_a_loss, arm_b_loss
from .losses import LossReport, VCCStage1Loss
from .sampler import BatchSpec, RectangularPertBatchSampler
from .synthetic import SyntheticFixture, build_default_fixture

__all__ = [
    "PertBatch",
    "PerturbationDataAdapter",
    "build_batch",
    "LossReport",
    "VCCStage1Loss",
    "VCCCompositeLoss",
    "arm_a_loss",
    "arm_b_loss",
    "BatchSpec",
    "RectangularPertBatchSampler",
    "SyntheticFixture",
    "build_default_fixture",
]
