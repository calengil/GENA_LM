#!/usr/bin/env python3
"""Prepare stock ``final_pert`` mapping files from existing VCC qref exports.

This does not renormalize qref values.  It adds the explicit non-targeting
profile required by ``expression_dataset_final_pert.py``, resolves target
symbols to interval Ensembl IDs, and writes one shared TF-panel gene order and
log1p-qnorm bin edges.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXTS = ("orion_hct116", "orion_hek293t", "k562_gwps")


def interval_table(intervals: Path, context: str) -> pd.DataFrame:
    parts = [
        pd.read_csv(intervals / f"{context}.{strand}.csv", sep=None, engine="python", dtype=str)
        for strand in ("forward", "reverse")
    ]
    table = pd.concat(parts, ignore_index=True)
    return table.drop_duplicates("gene_name", keep="first")


def write_profile(path: Path, gene_ids: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(gene_ids)
        writer.writerow(np.asarray(values, dtype=np.float32).tolist())


def tokenizer_vocab(path_or_name: str) -> dict:
    path = Path(path_or_name)
    tokenizer_json = path / "tokenizer.json"
    if tokenizer_json.is_file():
        payload = json.loads(tokenizer_json.read_text())
        return payload["model"]["vocab"]
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path_or_name).get_vocab()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--qref", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--tf-tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel_ids: list[str] = []
    seen: set[str] = set()
    controls_log1p: list[np.ndarray] = []
    audit = {"status": "PASS", "normalization": "existing qref; no renormalization", "contexts": {}}

    for context in CONTEXTS:
        table = interval_table(args.intervals, context)
        symbol_to_id = dict(zip(table["gene_name"], table["gene_id"]))
        with np.load(args.arrays / f"{context}.npz", allow_pickle=False) as data:
            symbols = data["panel_genes"].astype(str).tolist()
            targets = data["target_genes"].astype(str).tolist()
            control = data["control"].astype(np.float32)
        missing_panel = [gene for gene in symbols if gene not in symbol_to_id]
        if missing_panel:
            raise ValueError(f"{context}: panel symbols absent from intervals: {missing_panel[:20]}")
        gene_ids = [symbol_to_id[gene] for gene in symbols]
        for gene_id in gene_ids:
            if gene_id not in seen:
                seen.add(gene_id)
                panel_ids.append(gene_id)
        controls_log1p.append(np.log1p(control[np.isfinite(control)]))

        source_map = pd.read_csv(args.qref / context / "file_mappings.csv", dtype=str)
        if source_map["id"].tolist() != targets:
            raise ValueError(f"{context}: qref mapping order differs from NPZ target order")
        out_context = args.out / context
        out_context.mkdir(exist_ok=True)
        control_path = out_context / "non_targeting.csv"
        write_profile(control_path, gene_ids, control)

        rows = [{
            "id": f"{context}__non_targeting",
            "dataset": context,
            "perturbation": "non-targeting",
            "dataset_description": context,
            "tpm": control_path.name,
        }]
        unresolved = []
        for target, source_tpm in zip(targets, source_map["tpm"]):
            source_profile = (args.qref / context / source_tpm).resolve()
            relative_profile = os.path.relpath(source_profile, out_context.resolve())
            perturbation_id = symbol_to_id.get(target)
            if perturbation_id is None:
                unresolved.append(target)
                perturbation_id = target
            rows.append({
                "id": target,
                "dataset": context,
                "perturbation": perturbation_id,
                "dataset_description": context,
                "tpm": relative_profile,
            })
        pd.DataFrame(rows).to_csv(out_context / "file_mappings.csv", index=False)
        audit["contexts"][context] = {
            "targets": len(targets), "genes": len(symbols),
            "unresolved_target_symbols": unresolved,
        }

    (args.out / "panel_genes.txt").write_text("\n".join(panel_ids) + "\n")
    vocab = tokenizer_vocab(args.tf_tokenizer)
    n_bins = sum(f"[BIN_{i}]" in vocab for i in range(len(vocab)))
    if n_bins < 2:
        raise ValueError(f"no [BIN_i] vocabulary in {args.tf_tokenizer}")
    pooled = np.concatenate(controls_log1p)
    edges = np.quantile(pooled, np.arange(1, n_bins) / n_bins).astype(np.float32)
    np.save(args.out / "panel_bin_edges.npy", edges)
    audit["panel_genes"] = len(panel_ids)
    audit["tf_bins"] = n_bins
    audit["bin_space"] = "log1p(existing qref control)"
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
