#!/bin/bash
# Dual-Critic Blending for Cooperative Multi-Agent RL
# Usage:
#   bash scripts/run.sh                    # Run ALL experiments (40 runs)
#   bash scripts/run.sh --method ippo      # Run only IPPO
#   bash scripts/run.sh --method mappo     # Run only MAPPO
#   bash scripts/run.sh --method blend025  # Run only Blend α=0.25
#   bash scripts/run.sh --method blend050  # Run only Blend α=0.5
#   bash scripts/run.sh --method blend075  # Run only Blend α=0.75
#   bash scripts/run.sh --seeds 3          # Use 3 seeds instead of 8
#   bash scripts/run.sh --frames 500000    # Use 500K frames (quick test)
#   bash scripts/run.sh --dry-run          # Show what would run without running
#
# Examples:
#   bash scripts/run.sh --method ippo --seeds 2 --frames 100000   # Quick IPPO test
#   bash scripts/run.sh --method blend050 --seeds 8               # Full Blend α=0.5
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARL_DIR="$PROJECT_DIR/BenchMARL"
RESULTS_DIR="$PROJECT_DIR/results"

# Defaults
METHOD="all"
NUM_SEEDS=8
FRAMES=2000000
DRY_RUN=false
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)   METHOD="$2"; shift 2 ;;
        --seeds)    NUM_SEEDS="$2"; shift 2 ;;
        --frames)   FRAMES="$2"; shift 2 ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --help|-h)
            head -20 "$0" | grep "^#" | sed 's/^# *//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect GPU
if [ "$DEVICE" = "cuda" ]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "WARNING: CUDA not available, falling back to CPU"
        DEVICE="cpu"
    fi
fi

mkdir -p "$RESULTS_DIR"

# Common experiment arguments
COMMON_ARGS=(
    "experiment.max_n_frames=$FRAMES"
    "experiment.sampling_device=$DEVICE"
    "experiment.train_device=$DEVICE"
    "experiment.buffer_device=$DEVICE"
    "experiment.loggers=[csv]"
    "experiment.create_json=true"
    "experiment.checkpoint_at_end=true"
    "experiment.save_folder=$RESULTS_DIR"
    "experiment.render=false"
    "experiment.evaluation=true"
    "experiment.evaluation_interval=120000"
    "experiment.clip_grad_val=1.0"
    "algorithm.use_tanh_normal=False"
)

TASK="pettingzoo/multiwalker"

# Define experiment configurations
declare -A METHODS
METHODS[ippo]="algorithm=ippo"
METHODS[mappo]="algorithm=mappo"
METHODS[blend025]="algorithm=progressive algorithm.alpha_start=0.25 algorithm.alpha_end=0.25 algorithm.anneal_schedule=constant"
METHODS[blend050]="algorithm=progressive algorithm.alpha_start=0.5 algorithm.alpha_end=0.5 algorithm.anneal_schedule=constant"
METHODS[blend075]="algorithm=progressive algorithm.alpha_start=0.75 algorithm.alpha_end=0.75 algorithm.anneal_schedule=constant"

# Determine which methods to run
if [ "$METHOD" = "all" ]; then
    RUN_METHODS=(ippo mappo blend025 blend050 blend075)
else
    if [ -z "${METHODS[$METHOD]}" ]; then
        echo "ERROR: Unknown method '$METHOD'"
        echo "Available: ippo, mappo, blend025, blend050, blend075, all"
        exit 1
    fi
    RUN_METHODS=("$METHOD")
fi

# Generate seed list
SEEDS=()
for ((i=0; i<NUM_SEEDS; i++)); do
    SEEDS+=("$i")
done

# Count and display plan
TOTAL=0
for m in "${RUN_METHODS[@]}"; do
    TOTAL=$((TOTAL + NUM_SEEDS))
done

echo "#################### Dual-Critic Blending ####################"
echo "Task:     $TASK"
echo "Frames:   $FRAMES"
echo "Device:   $DEVICE"
echo "Seeds:    ${SEEDS[*]}"
echo "Methods:  ${RUN_METHODS[*]}"
echo "Total:    $TOTAL runs"
echo "Est time: ~$((TOTAL * 75 / 60)) hours (at ~75 min/run on GPU)"
echo "Results:  $RESULTS_DIR"
echo "##############################################################"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute:"
    for m in "${RUN_METHODS[@]}"; do
        for s in "${SEEDS[@]}"; do
            echo "  python benchmarl/run.py ${METHODS[$m]} task=$TASK seed=$s ..."
        done
    done
    exit 0
fi

# Run experiments
COMPLETED=0
FAILED=0

for m in "${RUN_METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        RUN_NAME="${m}_s${s}"
        START_TIME=$(date +%s)
        echo ""
        echo "[$(date +%H:%M:%S)] Starting: $RUN_NAME ($((COMPLETED + FAILED + 1))/$TOTAL)"

        # Build command
        CMD="python benchmarl/run.py ${METHODS[$m]} task=$TASK seed=$s ${COMMON_ARGS[*]}"

        if timeout 7200 bash -c "cd '$BENCHMARL_DIR' && $CMD" 2>&1 | tail -1; then
            ELAPSED=$(( $(date +%s) - START_TIME ))
            echo "[$(date +%H:%M:%S)] $RUN_NAME: OK (${ELAPSED}s)"
            COMPLETED=$((COMPLETED + 1))
        else
            ELAPSED=$(( $(date +%s) - START_TIME ))
            echo "[$(date +%H:%M:%S)] $RUN_NAME: FAILED (${ELAPSED}s)"
            FAILED=$((FAILED + 1))
        fi
    done
done

# Summary
echo ""
echo "##############################################################"
echo "COMPLETE: $COMPLETED/$TOTAL succeeded, $FAILED failed"
echo "Results:  $RESULTS_DIR"
echo "##############################################################"
