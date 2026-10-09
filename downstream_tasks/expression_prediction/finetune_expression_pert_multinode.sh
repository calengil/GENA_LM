#!/usr/bin/env bash
# Training of panel_pert on several nodes. The same script is started on every node;
# only MACHINE_RANK differs (0 on the main node). Example for 2 nodes:
#   node 0: NUM_MACHINES=2 MACHINE_RANK=0 MAIN_IP=<ip of node 0> ./finetune_expression_pert_multinode.sh
#   node 1: NUM_MACHINES=2 MACHINE_RANK=1 MAIN_IP=<ip of node 0> ./finetune_expression_pert_multinode.sh
# The global main process (node 0, rank 0) tokenizes the genes and reads the expression data
# into the caches; every other process waits for it at a barrier and then reads the caches.
set -e

cd ../..

export CUDA_HOME="$HOME/.local/cuda/"
export PATH="$HOME/.local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/cuda/lib64/:$LD_LIBRARY_PATH"

NUM_MACHINES=${NUM_MACHINES:?set NUM_MACHINES, the number of nodes}
MACHINE_RANK=${MACHINE_RANK:?set MACHINE_RANK, the number of this node: 0 .. NUM_MACHINES-1}
MAIN_IP=${MAIN_IP:?set MAIN_IP, the address of node 0 reachable from every node}
MAIN_PORT=${MAIN_PORT:-29515}   # one port per run: two runs on the same nodes need different ports

# how long the processes wait for each other (minutes): must cover building the caches from scratch;
# it is also how long a real hang of a collective takes to be reported
export DIST_TIMEOUT_MINUTES=${DIST_TIMEOUT_MINUTES:-80}

# GPUs of this node; every node must have the same number
export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7
GPUS_PER_NODE=4

# export NCCL_DEBUG=INFO                 # when the start hangs at the process group initialisation
# export NCCL_SOCKET_IFNAME=<interface>  # when NCCL picks a network interface the nodes cannot reach

TBS=80
BS=1
NP=$(( NUM_MACHINES * GPUS_PER_NODE ))  # processes of all nodes
if (( TBS % (BS * NP) != 0 )); then
  echo "TBS=$TBS is not divisible by BS*NP=$((BS * NP)): the global batch would be smaller" >&2
  exit 1
fi
GAS=$(( TBS / (BS * NP) ))

config_name="final_pert_multinode"

GENALM_HOME=$(realpath ..) accelerate launch \
  --multi_gpu \
  --num_machines "$NUM_MACHINES" \
  --machine_rank "$MACHINE_RANK" \
  --main_process_ip "$MAIN_IP" \
  --main_process_port "$MAIN_PORT" \
  --num_processes "$NP" \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final_pert_multinode \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS"

echo "done"
