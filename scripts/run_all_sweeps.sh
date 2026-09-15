#!/usr/bin/env bash
# Full sweep across all three datasets and all three index families.
set -u
cd "$(dirname "$0")/.."
for ds in sift-128-euclidean glove-100-angular nytimes-256-angular; do
  echo "=== $ds ==="
  python3 src/sweep.py --dataset "$ds" --n-base 200000 --n-query 300 --k 10 --repeats 3
done
echo "ALL SWEEPS DONE"
