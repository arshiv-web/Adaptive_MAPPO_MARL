#!/bin/bash

ACCOUNTS=(
  ece567w26_class
)

ALGOS=("adaptive_mappo" "mappo" "ippo" "fixed_alpha_mappo")


TASKS=(
  "vmas/navigation"
  "vmas/balance"
  "pettingzoo/multiwalker"
  "pettingzoo/simple_world_comm"
)

SEEDS=(0 1 2)

i=0
for algo in "${ALGOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      account=${ACCOUNTS[$((i % ${#ACCOUNTS[@]}))]}
      task_tag=$(echo "$task" | tr '/' '_')
      if [[ "$ALGO" == "mappo" || "$ALGO" == "ippo" || "$ALGO" == "adaptive_mappo" ]]; then
        sbatch \
          --account="$account" \
          --job-name=rl1m_${task_tag}_${algo}_s${seed} \
          --export=ALL,ACCOUNT="$account",ALGO="$algo",TASK="$task",TASK_TAG="$task_tag",SEED="$seed" \
          run_2m_single.sh
      else
        sbatch \
          --account="$account" \
          --job-name=rl1m_${task_tag}_${algo}_s${seed} \
          --export=ALL,ACCOUNT="$account",ALGO="$algo",TASK="$task",TASK_TAG="$task_tag",SEED="$seed",ALPHA="0.25" \
          run_2m_single.sh
        
        sbatch \
          --account="$account" \
          --job-name=rl1m_${task_tag}_${algo}_s${seed} \
          --export=ALL,ACCOUNT="$account",ALGO="$algo",TASK="$task",TASK_TAG="$task_tag",SEED="$seed",ALPHA="0.50" \
          run_2m_single.sh

        sbatch \
          --account="$account" \
          --job-name=rl1m_${task_tag}_${algo}_s${seed} \
          --export=ALL,ACCOUNT="$account",ALGO="$algo",TASK="$task",TASK_TAG="$task_tag",SEED="$seed",ALPHA="0.75" \
          run_2m_single.sh
      fi
      echo "Submitted 2M OFFLINE: $task $algo seed=$seed on $account"
      i=$((i + 1))
    done
  done
done
