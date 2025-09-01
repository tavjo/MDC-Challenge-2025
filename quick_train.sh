#!/bin/bash

# Quick Training Script - Simple wrapper for basic Random Forest training
# This is a minimal version for quick testing

set -e

echo "🚀 Starting Quick Random Forest Training..."
echo "==========================================="

# Check if training data exists
if [[ ! -f "Data/train/train_data.csv" ]]; then
    echo "❌ Error: Training data not found at Data/train/train_data.csv"
    exit 1
fi

# Create output directory
mkdir -p artifacts/models

# Run training with default parameters
echo "📊 Training Random Forest with default parameters..."
echo "   - Input: Data/train/train_data.csv"
echo "   - Output: artifacts/models"
echo "   - Target: target"
echo "   - Seed: 42"
echo "   - Iterations: 500"
echo "   - No holdout"
echo ""

python src/training.py \
    --input_csv Data/train/train_data.csv \
    --output_dir artifacts/models \
    --target_col target \
    --seed 42 \
    --n_iter 500 \
    --no_holdout

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📁 Check artifacts/models/ for results:"
    ls -la artifacts/models/
else
    echo "❌ Training failed!"
    exit 1
fi