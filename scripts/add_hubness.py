"""Add hubness (k-occurrence skewness, k = 10) to the dataset characteristics,
computed on the same 20,000-vector sample used for the other features."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import datasets, characterize

path = os.path.join(ROOT, "results", "dataset_characteristics.json")
rows = json.load(open(path))
for r in rows:
    base, q, metric, _ = datasets.load(r["dataset"], n_base=200000, n_query=300)
    rng = np.random.default_rng(0)
    idx = rng.choice(base.shape[0], size=min(20000, base.shape[0]), replace=False)
    r["hubness_skew_k10"] = characterize.hubness(np.ascontiguousarray(base[idx]), metric)
    print(f'{r["dataset"]:30s} hubness {r["hubness_skew_k10"]:.2f}', flush=True)
json.dump(rows, open(path, "w"), indent=2)
