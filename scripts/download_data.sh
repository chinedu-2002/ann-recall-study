#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../data"
for f in sift-128-euclidean glove-100-angular nytimes-256-angular; do
  [ -f "$f.hdf5" ] || curl -sS -o "$f.hdf5" "http://ann-benchmarks.com/$f.hdf5"
done
