#!/bin/bash

ACCOUNTS=(
  ece567w26_class
)

ALGOS=("mappo" "ippo")

TASKS=(
  "vmas/navigation"
  "vmas/balance"
  "pettingzoo/multiwalker"
  "pettingzoo/simple_tag"
)

SEEDS=(0 1 2)

i=0
for task in "${TASKS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      account=${ACCOUNTS[$((i % ${#ACCOUNTS[@]}))]}
      task_tag=$(echo "$task" | tr '/' '_')

      sbatch \
        --account="$account" \
        --job-name=rl1m_${task_tag}_${algo}_s${seed} \
        --export=ALL,ACCOUNT="$account",ALGO="$algo",TASK="$task",TASK_TAG="$task_tag",SEED="$seed" \
        run_2m_single.sh

      echo "Submitted 2M OFFLINE: $task $algo seed=$seed on $account"
      i=$((i + 1))
    done
  done
done