# Composite-loss training

The upstream `_pert` files are unchanged. The VCC run uses five parallel entrypoints:

1. `expression_dataset_final_pert_vcc.py` reads audited context NPZ files and
   delegates DNA tokenization and TF-panel construction to the stock dataset.
2. `expression_model_final_pert_vcc.py` subclasses the stock model, restores a
   complete gene-by-perturbation rectangle and evaluates the composite loss.
3. `run_expression_finetuning_final_pert_vcc.py` uses context-pure rectangular
   batches. Every rank sees the same response-gene batch and computes a different perturbation shard.
4. `finetune_expression_pert_vcc.sh` is the only launch command.
5. `configs/final_pert_vcc.yaml` selects Arm B and the bundled datasets.

Arm B is `Huber + 0.1*PDS + 0.02*NMAE + 0.1*FID-v2A + 0.0333*Reach-v2A + 0.05*Jaccard`.
The target-gene response row is excluded where required. Predictions and matched controls
are converted to LFC inside the differentiable loss. H1 is validation-only.

## Launch

Set `GENALM_HOME` to the directory containing `GENA_LM`, `models`, `genomes` and `runs`, then:

```bash
cd "$GENALM_HOME/GENA_LM/downstream_tasks/expression_prediction"
CUDA_VISIBLE_DEVICES=0 NP=1 BS=1 TBS=8 bash finetune_expression_pert_vcc.sh
```

For four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 BS=1 TBS=8 bash finetune_expression_pert_vcc.sh
```

`micro_perturbations=4` is the validated default. `BS` is the number of response
genes per context-pure batch; `TBS` controls gradient accumulation. Do not use
the stock `finetune_expression_pert.sh` for the composite run.
