#!/bin/bash
# Install BenchMARL + dual-critic blending algorithm
# Usage:
#       bash scripts/install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Installation......"
echo ""


# 1. Clone BenchMARL
if [ ! -d "$PROJECT_DIR/BenchMARL" ]; then
    echo "[1/4] Cloning BenchMARL..."
    git clone https://github.com/facebookresearch/BenchMARL.git "$PROJECT_DIR/BenchMARL"
else
    echo "[1/4] BenchMARL already exists, skipping clone"
fi


# 2. Create virtual environment
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "[2/4] Creating virtual environment..."
    if command -v uv &>/dev/null; then
        uv venv "$PROJECT_DIR/.venv" --python 3.11
    else
        python3 -m venv "$PROJECT_DIR/.venv"
    fi
else
    echo "[2/4] Virtual environment already exists"
fi
source "$PROJECT_DIR/.venv/bin/activate"


# 3. Install dependencies
echo "[3/4] Installing dependencies..."
if command -v uv &>/dev/null; then
    uv pip install -e "$PROJECT_DIR/BenchMARL"
    uv pip install "pettingzoo[sisl]" matplotlib scipy pyyaml
else
    pip install -e "$PROJECT_DIR/BenchMARL"
    pip install "pettingzoo[sisl]" matplotlib scipy pyyaml
fi


# 4. Install algorithm into BenchMARL
echo "[4/4] Installing dual-critic blending algorithm..."

ALGO_DIR="$PROJECT_DIR/BenchMARL/benchmarl/algorithms"
CONF_DIR="$PROJECT_DIR/BenchMARL/benchmarl/conf/algorithm"

cp "$PROJECT_DIR/algorithms/dual_critic_blend.py" "$ALGO_DIR/progressive.py"
cp "$PROJECT_DIR/configs/dual_critic_blend.yaml" "$CONF_DIR/progressive.yaml"

# Patch __init__.py to register our algorithm (if not already registered)
if ! grep -q "progressive" "$ALGO_DIR/__init__.py"; then
    # Add import
    sed -i '/from .mappo import/a from .progressive import Progressive, ProgressiveConfig' "$ALGO_DIR/__init__.py"
    # Add to classes list
    sed -i '/"MasacConfig",/a\    "Progressive",\n    "ProgressiveConfig",' "$ALGO_DIR/__init__.py"
    # Add to registry
    sed -i '/"iql": IqlConfig,/a\    "progressive": ProgressiveConfig,' "$ALGO_DIR/__init__.py"
    echo "  Registered 'progressive' algorithm in BenchMARL"
else
    echo "  Algorithm already registered"
fi

echo ""
echo "Installation complete!!!!!!!!!!!!!!!!"
echo "Activate environment: source $PROJECT_DIR/.venv/bin/activate"
echo "Run experiments:      bash scripts/run.sh --help"
