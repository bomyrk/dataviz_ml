#!/bin/bash
set -e

# Paths relative to project root
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Activate venv if present
[ -f venv/bin/activate ] && source venv/bin/activate

# Train
# python src/modeling/train.py --data data/processed/mnist_train_folds.csv
python src/modeling/train.py --fold 0
python src/modeling/train.py --fold 1
python src/modeling/train.py --fold 2
python src/modeling/train.py --fold 3
python src/modeling/train.py --fold 4

echo "Done: $(date)"   
