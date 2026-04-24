#!/bin/bash
# scripts/train_cloud.sh
# Run this on RunPod/Lambda after cloning the repo.

set -e

echo "=== Setting up environment ==="
pip install -q -r requirements.txt
pip install -q taichi  # soft body sim, separate install

echo "=== Generating synthetic data ==="
python -m src.data.generate_synthetic

echo "=== Running smoke test ==="
python scripts/smoke_test.py

echo "=== Starting training ==="
# Set your wandb key: export WANDB_API_KEY=your_key
python -m src.train \
  training.batch_size=32 \
  training.num_workers=8 \
  training.max_epochs=100 \
  wandb.enabled=true

# To resume: RESUME_CHECKPOINT=outputs/checkpoints/epoch_0050.pt python -m src.train
