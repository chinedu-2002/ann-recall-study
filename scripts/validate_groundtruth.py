"""Harness validation gate 1: does our brute-force ground truth agree with the
published ground truth shipped in the ANN-Benchmarks HDF5 files?

If it does not, every recall number downstream is meaningless, so this runs
first and its result is the gate for trusting the rest of the study.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import datasets, groundtruth

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
rows = []
for name in ["sift-128-euclidean", "glove-100-angular", "nytimes-256-angular"]:
    base, q, metric, published = datasets.load(name, n_base=None, n_query=1000)
    t0 = time.perf_counter()
    ours, _ = groundtruth.exact_knn(base, q, 100, metric)
    el = time.perf_counter() - t0
    agree = {}
    for k in (1, 10, 100):
        agree[f"agreement@{k}"] = groundtruth.recall_at_k(ours, published, k)
    row = {"dataset": name, "n_base": int(base.shape[0]), "dim": int(base.shape[1]),
           "metric": metric, "n_queries": int(q.shape[0]),
           "bruteforce_seconds": round(el, 2), **agree}
    print(json.dumps(row))
    rows.append(row)
    np.save(os.path.join(OUT, f"gt_full_{name}.npy"), ours)
with open(os.path.join(OUT, "groundtruth_validation.json"), "w") as f:
    json.dump(rows, f, indent=2)
