#!/bin/bash
set -euo pipefail

MODEL_DIR=${1:-models/bert-sentiment}
DATA_DIR=${2:-datasets/sentiment}

echo "Evaluating model: $MODEL_DIR"
python3 -m evaluate --model $MODEL_DIR --data $DATA_DIR/test.csv --metrics accuracy f1
