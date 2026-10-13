#!/usr/bin/env bash
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

TBS=${TBS:-8}
BS=${BS:-1}
NP=${NP:-1}
if (( TBS % BS != 0 )); then
  echo "TBS must be divisible by BS" >&2
  exit 2
fi
# Every rank sees the same G batch, so NP does not multiply effective G.
GAS=$(( TBS / BS ))

config_name="final_pert_vcc"

launch_args=(--main_process_port 29515 --num_processes "$NP")
if (( NP > 1 )); then
  launch_args+=(--multi_gpu)
fi

GENALM_HOME=$(realpath ..) accelerate launch "${launch_args[@]}" \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final_pert_vcc \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"
