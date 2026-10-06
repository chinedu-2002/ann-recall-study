#!/usr/bin/env bash
# Progress Report 2 experiment pipeline, in priority order. Runs sequentially so
# that no two jobs compete for the two cores while latency is being timed.
set -u
cd "$(dirname "$0")/.."
export ANN_BUILD_THREADS=2
REAL="sift-128-euclidean glove-100-angular nytimes-256-angular fashion-mnist-784-euclidean mnist-784-euclidean glove-25-angular"
SYN="synthetic-id4-128 synthetic-id8-128 synthetic-id16-128 synthetic-id24-128 synthetic-id32-128 synthetic-id48-128"

echo "=== STAGE 1: dense sweeps for the heuristic ($(date -u +%H:%M)) ==="
for ds in $REAL $SYN; do
  python3 src/sweep.py --dataset "$ds" --profile dense --n-base 200000 --n-query 300 --repeats 3
done

echo "=== STAGE 2: Progress Report 1 grid on the three new real datasets ($(date -u +%H:%M)) ==="
for ds in fashion-mnist-784-euclidean mnist-784-euclidean glove-25-angular; do
  python3 src/sweep.py --dataset "$ds" --profile pr1 --n-base 200000 --n-query 300 --repeats 3
done

echo "=== STAGE 3: k = 1 and k = 100 passes ($(date -u +%H:%M)) ==="
for ds in sift-128-euclidean glove-100-angular nytimes-256-angular; do
  python3 src/sweep.py --dataset "$ds" --profile kpass --n-base 200000 --n-query 300 --repeats 3
done

echo "=== STAGE 4: full-size base sets ($(date -u +%H:%M)) ==="
for ds in sift-128-euclidean glove-100-angular; do
  python3 src/sweep.py --dataset "$ds" --profile scale --n-base 0 --n-query 300 --repeats 3
done
echo "ALL PR2 EXPERIMENTS DONE ($(date -u +%H:%M))"
