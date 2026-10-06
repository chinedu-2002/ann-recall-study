"""Confidence intervals for the Progress Report 1 default configurations.

Those runs predate per-query recall logging. Single-threaded builds are
deterministic, so rebuilding the three default configurations reproduces the
exact indexes measured in Progress Report 1, and this time the spread of
per-query recall is kept so each default recall gets a 95% interval."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["ANN_BUILD_THREADS"] = "1"
import numpy as np
import datasets, groundtruth, indexes
from sweep import dists_to, per_query_tieaware, _snap_m

rows = []
for name in ["sift-128-euclidean", "glove-100-angular", "nytimes-256-angular"]:
    base, q, metric, _ = datasets.load(name, n_base=200000, n_query=300)
    _, raw = groundtruth.exact_knn(base, q, 10, metric)
    td = (1.0 - raw) if metric == "angular" else np.sqrt(np.maximum(raw, 0))
    dim = base.shape[1]
    for fam, ix, qp in (("HNSW", indexes.HNSWIndex(dim, metric, M=16, efConstruction=200), 10),
                        ("IVF-Flat", indexes.IVFFlatIndex(dim, metric, nlist=100), 1),
                        ("IVF-PQ", indexes.IVFPQIndex(dim, metric, nlist=1024, m=_snap_m(8, dim), nbits=8), 1)):
        ix.build(base); ix.set_query_param(qp)
        found, _ = ix.search(q, 10)
        pq = per_query_tieaware(dists_to(q, base, found.astype(np.int64), metric), td, 10)
        r = {"dataset": name, "family": fam, "recall": round(float(pq.mean()), 5),
             "q_std": round(float(pq.std(ddof=1)), 5), "n": int(len(pq)),
             "ci95": round(1.96 * float(pq.std(ddof=1)) / np.sqrt(len(pq)), 5)}
        rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(os.path.join(ROOT, "results", "pr1_default_ci.json"), "w"), indent=2)
