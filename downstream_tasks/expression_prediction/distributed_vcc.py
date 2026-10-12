"""Exact perturbation-axis sharding for the coupled VCC composite loss."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.utils.checkpoint import checkpoint


def distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def perturbation_shard(total_p: int) -> tuple[int, int, int]:
    """Return ``(start, end, padded_width)`` for the current rank."""
    if total_p < 1:
        raise ValueError("total_p must be positive")
    if not distributed_enabled():
        return 0, total_p, total_p
    world_size = dist.get_world_size()
    if world_size > total_p:
        raise ValueError(
            f"world_size={world_size} cannot exceed perturbations P={total_p}"
        )
    width = math.ceil(total_p / world_size)
    start = dist.get_rank() * width
    end = min(total_p, start + width)
    return start, end, width


def gather_perturbation_axis(
    local_predictions: torch.Tensor,
    total_p: int,
) -> torch.Tensor:
    """Differentiably reconstruct ``[G, P, ...]`` on every rank."""
    if local_predictions.ndim < 2:
        raise ValueError("local_predictions must have shape [G, local_P, ...]")
    if not distributed_enabled():
        return local_predictions[:, :total_p]
    gathered = all_gather(local_predictions.contiguous())
    return torch.cat(tuple(gathered), dim=1)[:, :total_p]


def distributed_model_forward(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tf_ids: torch.Tensor,
    tf_attention_mask: torch.Tensor,
    dataset_flag: torch.Tensor,
    micro_p: int,
) -> torch.Tensor:
    """Compute local P shards and return full logits ``[G, P, L, 1]``."""
    if dataset_flag.ndim != 2:
        raise ValueError("dataset_flag must have shape [G, P]")
    if micro_p < 1:
        raise ValueError("micro_p must be positive")
    genes, perturbations = dataset_flag.shape
    for name, value in (
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
        ("tf_ids", tf_ids),
        ("tf_attention_mask", tf_attention_mask),
    ):
        if value.shape[:2] != (genes, perturbations):
            raise ValueError(
                f"{name} starts with {tuple(value.shape[:2])}; "
                f"expected {(genes, perturbations)}"
            )

    start, end, padded_width = perturbation_shard(perturbations)
    local_chunks = []
    for chunk_start in range(start, end, micro_p):
        chunk_end = min(end, chunk_start + micro_p)
        chunk_width = chunk_end - chunk_start

        def run_chunk(ids, mask, panel, panel_mask, flags):
            return model.forward_base(
                input_ids=ids,
                attention_mask=mask,
                tf_ids=panel,
                tf_attention_mask=panel_mask,
                dataset_flag=flags,
                labels=None,
                return_dict=True,
            ).logits

        args = (
            input_ids[:, chunk_start:chunk_end],
            attention_mask[:, chunk_start:chunk_end],
            tf_ids[:, chunk_start:chunk_end],
            tf_attention_mask[:, chunk_start:chunk_end],
            dataset_flag[:, chunk_start:chunk_end],
        )
        if model.training and torch.is_grad_enabled():
            logits = checkpoint(run_chunk, *args, use_reentrant=False)
        else:
            logits = run_chunk(*args)
        local_chunks.append(
            logits.reshape(genes, chunk_width, *logits.shape[1:])
        )

    local_logits = torch.cat(local_chunks, dim=1)
    local_width = end - start
    if local_width < padded_width:
        padding = local_logits.new_zeros(
            (genes, padded_width - local_width, *local_logits.shape[2:])
        )
        local_logits = torch.cat((local_logits, padding), dim=1)
    return gather_perturbation_axis(local_logits, perturbations)
