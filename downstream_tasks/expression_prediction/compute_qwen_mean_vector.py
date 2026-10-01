"""
Offline step before training: the mean of the condition vectors that the old Qwen
description model added to the DNA tokens.

For every track of every train_dataset*/valid_dataset*/test_dataset* table of the
experiment config: description text built from its metadata JSON (the same builder the
checkpoint was trained with) -> desc_model -> last token -> desc_proj -> desc_ln,
using the desc_* weights of args_params.init_checkpoint. The mean over all tracks is
saved to model_kwargs.qwen_mean_path.

Run from the GENA_LM root (finetune_expression_tf_qwen.sh does it):
    GENALM_HOME=... python -m downstream_tasks.expression_prediction.compute_qwen_mean_vector \
        --experiment_config downstream_tasks/expression_prediction/configs/final_tf_qwen.yaml
"""

import argparse
import json
import os

import pandas as pd
from omegaconf import OmegaConf

SPLIT_PREFIXES = ("train_dataset", "valid_dataset", "test_dataset")
DESC_PREFIXES = ("desc_model.", "desc_proj.", "desc_ln.")


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def collect_descriptions(cfg):
    """{track id: description text} over the tables of all splits."""
    from downstream_tasks.expression_prediction.expression_dataset_final import ExpressionDataset

    texts = {}
    for key, block in cfg.items():
        if not str(key).startswith(SPLIT_PREFIXES):
            continue
        targets_path = block["targets_path"]
        base_dir = os.path.dirname(targets_path)
        df = pd.read_csv(targets_path)
        for _, row in df.iterrows():
            meta_path = row["metadata"]
            if not os.path.isabs(meta_path):
                meta_path = os.path.join(base_dir, meta_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # the original (not FixedDescriptionsMixin) builder: the checkpoint was trained on its texts
            text = ExpressionDataset.make_description_from_json(meta, row["id"], meta_path)
            if texts.get(row["id"], text) != text:
                raise ValueError(f"Track {row['id']} has different descriptions in different tables")
            texts[row["id"]] = text
        print(f"{key}: {len(df)} tracks from {targets_path}")
    if not texts:
        raise ValueError("No train_dataset*/valid_dataset*/test_dataset* tables in the config")
    return texts


def build_qwen_condition(desc_model_name, checkpoint_path):
    """desc_model -> last token -> desc_proj -> desc_ln, as in expression_model_final.ExpressionCounts."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class QwenCondition(nn.Module):
        def __init__(self, gen_hidden_size):
            super().__init__()
            # module names match the checkpoint keys desc_model.* / desc_proj.* / desc_ln.*
            self.desc_model = AutoModel.from_pretrained(
                desc_model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            self.desc_proj = nn.Linear(self.desc_model.config.hidden_size, gen_hidden_size)
            self.desc_ln = nn.LayerNorm(gen_hidden_size)

        def forward(self, input_ids, attention_mask):
            hidden = self.desc_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
            return self.desc_ln(self.desc_proj(hidden[:, -1]))

    state = torch.load(checkpoint_path, map_location="cpu")
    desc_state = {k: v for k, v in state.items() if k.startswith(DESC_PREFIXES)}
    if "desc_proj.weight" not in desc_state:
        raise ValueError(f"No desc_* weights in {checkpoint_path}")
    del state

    model = QwenCondition(gen_hidden_size=desc_state["desc_proj.weight"].shape[0])
    # strict: every weight of the Qwen part must come from the checkpoint, including the
    # fine-tuned last desc_model blocks, otherwise the vectors differ from the trained ones
    model.load_state_dict(desc_state, strict=True)
    print(f"loaded {len(desc_state)} desc_* tensors from {checkpoint_path}")
    return model


def compute_mean(texts, desc_model_name, text_max_seq_len, checkpoint_path):
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required: the Qwen model runs with flash_attention_2")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(desc_model_name, padding_side="left")
    model = build_qwen_condition(desc_model_name, checkpoint_path).to(device).eval()

    total = None
    n_truncated = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for text in texts.values():
            # one description at a time, tokenized exactly as in ExpressionDataset.precompute_descriptions
            enc = tokenizer(text, padding=False, truncation=True, max_length=text_max_seq_len, return_tensors="pt")
            n_truncated += int(enc["input_ids"].shape[1] >= text_max_seq_len)
            vector = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))[0].double().cpu()
            total = vector if total is None else total + vector

    mean = (total / len(texts)).float()
    if not torch.isfinite(mean).all():
        raise ValueError("The mean vector has non-finite values")
    print(f"{len(texts)} descriptions, {n_truncated} truncated to {text_max_seq_len} tokens, "
          f"mean vector: shape {tuple(mean.shape)}, norm {mean.norm():.4f}")
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--print-output-path", action="store_true",
                        help="only print model_kwargs.qwen_mean_path and exit")
    args = parser.parse_args()

    cfg = load_config(args.experiment_config)
    output_path = cfg["model_kwargs"]["qwen_mean_path"]
    if args.print_output_path:
        print(output_path)
        return
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists; delete it to recompute")

    params = cfg["qwen_mean_params"]
    texts = collect_descriptions(cfg)
    mean = compute_mean(texts, params["desc_model_name"], int(params["text_max_seq_len"]),
                        cfg["args_params"]["init_checkpoint"])

    import torch
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = f"{output_path}.{os.getpid()}.temp"
    torch.save(mean, temp_path)
    os.replace(temp_path, output_path)
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
