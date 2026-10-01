#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#  --multi_gpu \

TBS=80
BS=2
NP=4
GAS=$(( TBS / (BS * NP) ))  

config_name="final_tf_qwen"
config_path="downstream_tasks/expression_prediction/configs/${config_name}.yaml"

export GENALM_HOME=$(realpath ..)

# Step 1: mean Qwen condition vector; skipped when it already lies at model_kwargs.qwen_mean_path
qwen_mean_path=$(python -m downstream_tasks.expression_prediction.compute_qwen_mean_vector \
  --experiment_config "$config_path" --print-output-path)
if [ -f "$qwen_mean_path" ]; then
  echo "Qwen mean vector found at $qwen_mean_path, skipping its computation"
else
  echo "Computing the Qwen mean vector -> $qwen_mean_path"
  python -m downstream_tasks.expression_prediction.compute_qwen_mean_vector \
    --experiment_config "$config_path"
fi

# Step 2: training
accelerate launch \
  --main_process_port 29515 \
  --num_processes "$NP" \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final_tf_qwen \
  --experiment_config "$config_path" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"
