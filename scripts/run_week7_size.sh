#!/usr/bin/env bash
# Week 7: separate collection size from LID. Dense sweeps of the default HNSW
# build and IVF-Flat nlist=1024 at five collection sizes on SIFT and GloVe-100.
# The 200,000-vector runs already exist from weeks 5 and 6.
set -u
cd "$(dirname "$0")/.."
export ANN_BUILD_THREADS=2
for ds in sift-128-euclidean glove-100-angular; do
  for n in 50000 100000 500000 0; do
    echo "=== $ds n=$n ($(date -u +%H:%M)) ==="
    python3 src/sweep.py --dataset "$ds" --profile dense --n-base "$n" --n-query 300 --repeats 3
  done
done
echo "WEEK7 SIZE SWEEP DONE ($(date -u +%H:%M))"
