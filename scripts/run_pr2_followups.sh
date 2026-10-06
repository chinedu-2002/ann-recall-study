#!/usr/bin/env bash
# Follow-up checks, run after the main pipeline and after build_variance.py.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "^python3 scripts/build_variance" >/dev/null; do sleep 10; done
python3 scripts/build_order.py nytimes-256-angular 1 2 3
python3 scripts/build_order.py sift-128-euclidean 1 2 3
python3 scripts/default_ci.py
echo FOLLOWUPS DONE
