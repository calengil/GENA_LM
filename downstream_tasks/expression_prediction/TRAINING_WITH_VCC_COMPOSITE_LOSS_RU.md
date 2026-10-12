# GENA-LM `_pert`: обучение на qref с VCC composite loss

## Что запускается

Штатный launcher — `run_expression_finetuning_final_pert.py`, базовая модель —
`expression_model_final_pert.py:ExpressionCounts`. Loss подключается через
`vcc_model_adapter.py:VCCExpressionCounts` и
`model_kwargs.vcc_loss`; encoder, TF encoder и decoder не заменяются.

- Arm A (контроль): `Huber + 0.1·PDS`.
- Arm B (приоритет): `Huber + 0.1·PDS + 0.02·NMAE + 0.1·FID-v2A +
  0.0333·Reach-v2A + 0.05·Jaccard`.

Готовый приоритетный конфиг: `configs/final_pert.yaml`. Arm A включается
отключением Stage-2 членов в том же блоке. v2-B — диагностика, v2-C исключён.

## Данные и их смысл

Канонические train-массивы в `data/vcc_context_arrays`:

| Контекст | P targets | G genes |
|---|---:|---:|
| Orion HCT116 | 300 | 20,343 |
| Orion HEK293T | 300 | 20,343 |
| K562 GWPS | 272 | 7,941 |

Каждый NPZ содержит `expression[P,G]`, `control[G]`, `lfc_true[P,G]`,
`sig_mask[P,G]`, `target_genes[P]`, `panel_genes[G]`, `target_gene_row[P]` и
`cell_count[P]`. Панели не урезаны до общего пересечения: каждый контекст
использует максимум доступных однозначно сопоставленных генов.

`expression/control` — qref-нормализованные transcript-span TPM. После qref
это уже не физические TPM, сумма не обязана быть 1e6. В NPZ они без `log1p`;
dataset одинаково применяет `log1p(x)` к label и control. В loss `expm1`
применяется только к этим модельным log1p-значениям для predicted LFC.

`lfc_true = log2((expression+1e-9)/(control+1e-9))`. `sig_mask` построена из
real single-cell данных: two-sided Mann–Whitney/Wilcoxon rank-sum на per-cell
`log1p(CPM)`, matched controls внутри sample/gem_group, BH отдельно для каждого
target по всей панели, alpha 0.05. Proxy-маски и нули не использованы.

H1 не входит в train. Это отдельная validation линия с 25 пересекающимися
targets. В пакет не включена выдуманная H1 DE-mask; полный Arm B loss на H1
без real `lfc_true/sig_mask` считать нельзя. H1 оценивается отдельным inference
pipeline после обучения.

## Адаптер к существующей архитектуре

Готовые qref profile CSV не пересчитываются. Команда ниже создаёт только
совместимые с `_pert` mapping-файлы, явную NTC-строку и общую panel schema:

```bash
python downstream_tasks/expression_prediction/prepare_vcc_pert_inputs.py \
  --arrays "$GENALM_HOME/data/vcc_context_arrays" \
  --qref "$GENALM_HOME/data/vcc_qref_export" \
  --intervals "$GENALM_HOME/intervals" \
  --tf-tokenizer "$GENALM_HOME/models/moderngena-tf-bins50" \
  --out "$GENALM_HOME/data/vcc_pert_inputs"
```

`VCCQrefExpressionDataset` наследует штатный dataset, поэтому сохраняет DNA
tokenization и TF-panel encoder, но фиксирует полный порядок P perturbations и
добавляет control/LFC/mask. Collate вычисляет `target_gene_row`; `-1` означает,
что target отсутствует в response panel.

`VCCContextBatchSampler` не смешивает контексты. Батч — прямоугольник
`G_chunk × P`, flatten order `gene_index * P + perturbation_index`.
`--batch_size` поэтому означает число генов `G_chunk`, не число profiles.

Hydra создаёт `VCCCompositeLoss` из YAML. `ExpressionCounts.forward()`
оставляет прежние logits, извлекает CLS prediction, собирает `(P,G_chunk)` и
вызывает loss. При `vcc_loss: null` остаётся legacy MSE. PDS сравнивает
направления perturbation delta (cosine InfoNCE) с исключением собственной
target-строки. NMAE/FID/Reach/Jaccard используют real LFC/mask. Это обучающие
дифференцируемые суррогаты, а не официальный `cell-eval2` scorer.

## Модели и checkpoint

В `models/` включены скачанные из `s3://genalm/phageml/virtualcell/models/`
decoder, ModernGENA TF model/tokenizer и
`all_datasets_2/pytorch_model.bin`. SHA-256 checkpoint:
`fffd4dabc782dbcee7c87a4b56e2c2f554b8bc1fdf2e0b7861cca4de7e752732`.

Для запуска нужен `pytorch_model_pert_init.bin`, собранный
`prepare_final_pert_checkpoint.py`: DNA/decoder/head берутся из
`all_datasets_2/pytorch_model.bin`, panel encoder — из
`moderngena-tf-bins50/model.safetensors`. Для resume используйте только полный
checkpoint, сохранённый этим `_pert` launcher.

## Внешний обязательный файл

GRCh38 FASTA локально не найден и в архив не подменён фиктивным файлом. Нужно
положить согласованные с intervals файлы в:

```text
genomes/GRCh38.primary_assembly.genome.fa
genomes/GRCh38.primary_assembly.genome.fa.fai
```

DNA tokenizer `AIRI-Institute/gena-lm-bert-base-t2t` должен быть в HF cache
или путь в YAML надо заменить локальным.

## Подготовка на новом сервере

```bash
tar -I zstd -xf vcc_tf_training_bundle_20261009.tar.zst
cd vcc_tf_training_bundle_20261009
export GENALM_HOME="$PWD"
sha256sum -c SHA256SUMS
python downstream_tasks/expression_prediction/prepare_vcc_pert_inputs.py \
  --arrays data/vcc_context_arrays --qref data/vcc_qref_export \
  --intervals intervals --tf-tokenizer models/moderngena-tf-bins50 \
  --out data/vcc_pert_inputs
export PYTHONPATH="$GENALM_HOME/GENA_LM:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

Нужна среда исходного GENA-LM плюс torch, transformers, hydra-core,
accelerate, numpy, pandas, scipy, scikit-learn и pyfaidx.

## Smoke и обучение

Сначала поставьте `args_params.iters: 20–50` в копии YAML и начните с
`--batch_size 8`, затем увеличьте до 16/32 по памяти:

```bash
accelerate launch --num_processes 1 --module \
  downstream_tasks.expression_prediction.run_expression_finetuning_final_pert \
  --experiment_config "$GENALM_HOME/GENA_LM/downstream_tasks/expression_prediction/configs/final_pert.yaml" \
  --batch_size 8 --gradient_accumulation_steps 1
```

Для Arm A замените YAML. Arm B — основной запуск, Arm A — контроль.

Все ranks получают одинаковые G-гены. Каждый rank считает свой непрерывный
shard по P, differentiable `all_gather` восстанавливает полную матрицу
`G_chunk × P`, и только затем считается composite. Внутри локального shard
память ограничивается `micro_perturbations` и activation checkpointing.
Стандартное значение — `micro_perturbations: 4`; для 20k-gene панели на
46-GB L40S может потребоваться уменьшение до 1.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 TBS=8 BS=8 \
  bash downstream_tasks/expression_prediction/finetune_expression_pert.sh
```

`NP` не умножает effective G, потому что батч реплицирован между ranks:
`G_effective = BS × gradient_accumulation_steps`. Число GPU не должно
превышать P минимального контекста.

Перед длинным run проверьте finite components/total, NaN/Inf, model gradients,
optimizer step, GPU memory/time и сохранение/загрузку полного `_pert` checkpoint.
Затем верните нужные `iters`, `save_interval` и output path.

Loss-пакет ранее прошёл 37/37 тестов и GPU loss-only smoke. После переноса на
правильные `_pert` файлы обязателен повторный model-level smoke
`forward→loss→backward→optimizer step`; это проверяет работоспособность, а не
качество модели.
