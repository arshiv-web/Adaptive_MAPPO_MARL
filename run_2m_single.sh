#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/.bashrc
conda activate benchmarl
module load cuda

mkdir -p logs

export WANDB_MODE=offline
export WANDB_PROJECT=benchmarl_phase1_2m
export WANDB_ENTITY=arshiv
export WANDB_DIR=/scratch/engin_root/engin1/arshiv/ml/wandb_2m
export WANDB_CACHE_DIR=/scratch/engin_root/engin1/arshiv/ml/wandb_cache_2m
export WANDB_CONFIG_DIR=/scratch/engin_root/engin1/arshiv/ml/wandb_config_2m

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "Running OFFLINE 2M: algo=$ALGO task=$TASK seed=$SEED account=$ACCOUNT"

SAVE_DIR=/scratch/engin_root/engin1/arshiv/ml/BenchMARL/runs/${TASK}/${ALGO}/seed_${SEED}
mkdir -p "$SAVE_DIR"

srun python benchmarl/run.py \
  algorithm=$ALGO \
  task=$TASK \
  seed=$SEED \
  experiment.max_n_frames=2000000 \
  experiment.sampling_device=cuda \
  experiment.train_device=cuda \
  experiment.buffer_device=cuda \
  experiment.save_folder="$SAVE_DIR" \
  experiment.lr=3e-4 \
  algorithm.entropy_coef=0.01