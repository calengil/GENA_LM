# VCC composite loss integration

`expression_model_final_pert.py` remains the stock architecture. The opt-in
`vcc_model_adapter.py` adds perturbation microbatching and `vcc_loss`. With
`model_kwargs.vcc_loss` in `final_pert.yaml` enables Arm B and replaces the model's training objective with
the composite over CLS expression predictions; sequence-token loss is not
added.

## Required batch contract

One batch contains one context and a complete rectangle of `G` response genes
by `P` perturbations. Rows are ordered `gene * P + perturbation`, so the
existing tensors have shapes:

- `dataset_flag`: `(G, P)`, all ones;
- `labels`, `labels_mask`: `(G, P, L, 1)`, with every CLS label present;
- `control_value`: `(G, P)` log1p(TPM), constant NTC value repeated over P;
- `target_gene_row`: `(G, P)` with the same `(P,)` vector repeated in each row;
- Arm B only: `lfc_true`, `sig_mask`: `(G, P)` from offline DE.

The model validates this contract and fails instead of silently accepting a
partial or shuffled rectangle. Dataset tensors are detached inside the model;
gradients flow only through predictions.

Every rank receives the same deterministic G batch and computes a contiguous
`P/world_size` perturbation shard. A differentiable `all_gather` restores the
complete `G x P` logits on every rank before the coupled loss. Activation
checkpointing further splits each local shard into perturbation microbatches.
The configured and adapter default is `micro_perturbations: 4`.
For identical full losses, gather-backward sums shard contributions and DDP
averaging recovers the single-GPU full-rectangle gradient.

## Arm A configuration

The ready configuration is `configs/final_pert.yaml`; its loss block is:

```yaml
  vcc_loss:
    _target_: downstream_tasks.expression_prediction.vcc_composite_loss.VCCCompositeLoss
    use_huber: true
    use_pds: true
    use_nmae: false
    use_fid: false
    use_reach: false
    use_jaccard: false
    huber_delta: 1.0
    pds_temperature: 0.1
    weight_huber: 1.0
    weight_pds: 0.1
```

Arm A is `1.0 * Huber + 0.1 * PDS`.

## Arm B configuration

```yaml
  vcc_loss:
    _target_: downstream_tasks.expression_prediction.vcc_composite_loss.VCCCompositeLoss
    use_huber: true
    use_pds: true
    use_nmae: true
    use_fid: true
    use_reach: true
    use_jaccard: true
    huber_delta: 1.0
    pds_temperature: 0.1
    weight_huber: 1.0
    weight_pds: 0.1
    weight_nmae: 0.02
    weight_fid: 0.1
    weight_reach: 0.0333
    weight_jaccard: 0.05
    theta: null
    tau_signif: 0.05
    tau_sign: 0.25
    eps: 1.0e-9
```

Arm B adds `0.02 * NMAE + 0.1 * FID + 0.0333 * Reach + 0.05 * Jaccard`.
`theta: null` calibrates the soft threshold from the offline DE mask.

## Launch

Use the repository's actual trainer, not the model file directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 TBS=8 BS=8 \
  bash downstream_tasks/expression_prediction/finetune_expression_pert.sh
```

Equivalent direct launch:

```bash
GENALM_HOME=/absolute/path/to/genalm accelerate launch \
  --multi_gpu --num_processes 4 \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final_pert \
  --experiment_config downstream_tasks/expression_prediction/configs/final_pert.yaml \
  --batch_size 8 \
  --gradient_accumulation_steps 1
```

Before launch, run `prepare_vcc_pert_inputs.py` to add the stock `_pert`
mapping columns/control row without renormalizing qref. Then use one optimizer
step and verify the logged `vcc_*_loss`
components, finite gradients, row ordering, and weighted gradient ratios on
the first real batch. The current context NPZ files provide real
`lfc_true` and Wilcoxon+BH `sig_mask` for Orion and K562.
