#!/usr/bin/env python3
"""Build an audited ``final_pert`` initialization checkpoint from the legacy model.

The published ``all_datasets_2/pytorch_model.bin`` predates the TF-panel
encoder.  Its DNA encoder, decoder and regression head are reusable, while the
old ``desc_model`` is not.  The TF-panel encoder is therefore taken from
``moderngena-tf-bins50/model.safetensors``.  The resulting state dict contains
only keys understood by ``expression_model_final_pert.ExpressionCounts``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


REUSED_PREFIXES = ("bert.", "decoder.", "dna_ln.", "classifier.")
LEGACY_PREFIXES = ("desc_model.", "desc_ln.", "desc_proj.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--tf-safetensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    legacy = torch.load(args.legacy, map_location="cpu", weights_only=True)
    if not isinstance(legacy, dict) or not legacy:
        raise TypeError("legacy checkpoint must be a non-empty state_dict")

    unknown = sorted(
        key for key in legacy
        if not key.startswith(REUSED_PREFIXES + LEGACY_PREFIXES)
    )
    if unknown:
        raise ValueError(f"unclassified legacy checkpoint keys: {unknown[:20]}")

    state = {
        key: value
        for key, value in legacy.items()
        if key.startswith(REUSED_PREFIXES)
    }
    reused_counts = {
        prefix.rstrip("."): sum(key.startswith(prefix) for key in state)
        for prefix in REUSED_PREFIXES
    }
    if reused_counts != {"bert": 170, "decoder": 20, "dna_ln": 2, "classifier": 2}:
        raise ValueError(f"unexpected reusable-key counts: {reused_counts}")

    tf_source = load_file(str(args.tf_safetensors), device="cpu")
    tf_body = {
        "tf." + key.removeprefix("model."): value
        for key, value in tf_source.items()
        if key.startswith("model.")
    }
    if len(tf_body) != 14:
        raise ValueError(f"expected 14 TF encoder keys, got {len(tf_body)}")
    state.update(tf_body)

    # Both are LayerNorms over the conditioning vector with the same width.
    # Reusing the trained normalization is preferable to silently random init.
    state["tf_ln.weight"] = legacy["desc_ln.weight"]
    state["tf_ln.bias"] = legacy["desc_ln.bias"]

    if len(state) != 210:
        raise ValueError(f"expected 210 final keys, got {len(state)}")
    if len(state) != len(set(state)):
        raise ValueError("duplicate output keys")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    report = {
        "status": "PASS",
        "legacy": str(args.legacy.resolve()),
        "legacy_sha256": sha256(args.legacy),
        "tf_safetensors": str(args.tf_safetensors.resolve()),
        "tf_safetensors_sha256": sha256(args.tf_safetensors),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "legacy_keys": len(legacy),
        "output_keys": len(state),
        "reused_counts": reused_counts,
        "tf_encoder_keys": len(tf_body),
        "tf_ln_source": "legacy desc_ln",
        "dropped_legacy_counts": {
            prefix.rstrip("."): sum(key.startswith(prefix) for key in legacy)
            for prefix in LEGACY_PREFIXES
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
