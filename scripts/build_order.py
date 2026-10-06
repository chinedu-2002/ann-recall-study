"""Is the rebuild effect caused by threading, or by insertion order?

Single-threaded hnswlib builds are deterministic, but they always insert in id
order. This builds the default HNSW configuration single-threaded with several
random insertion orders. If recall moves as much as it does between thread
counts, the cause is insertion order, and the single-threaded number from
Progress Report 1 is one draw from a distribution rather than a fixed property."""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, hnswlib
import datasets, groundtruth
from sweep import dists_to, per_query_tieaware

name = sys.argv[1] if len(sys.argv) > 1 else "nytimes-256-angular"
seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3]
base, q, metric, _ = datasets.load(name, n_base=200000, n_query=300)
tids, raw = groundtruth.exact_knn(base, q, 10, metric)
td = (1.0 - raw) if metric == "angular" else np.sqrt(np.maximum(raw, 0))
rows = []
for s in seeds:
    perm = np.random.default_rng(s).permutation(base.shape[0])
    ix = hnswlib.Index(space="l2" if metric == "euclidean" else "ip", dim=base.shape[1])
    ix.init_index(max_elements=base.shape[0], ef_construction=200, M=16, random_seed=42)
    ix.set_num_threads(1)
    t0 = time.perf_counter(); ix.add_items(base[perm], perm); bt = time.perf_counter() - t0
    ix.set_ef(10)
    ids, _ = ix.knn_query(q, k=10)
    ids = ids.astype(np.int64)
    rec = float(per_query_tieaware(dists_to(q, base, ids, metric), td, 10).mean())
    rows.append({"dataset": name, "build_threads": 1, "insertion_order_seed": s,
                 "recall_tieaware_ef10": round(rec, 5), "build_s": round(bt, 1)})
    print(json.dumps(rows[-1]), flush=True)
out = os.path.join(ROOT, "results", "build_order.json")
prev = json.load(open(out)) if os.path.exists(out) else []
json.dump([r for r in prev if r["dataset"] != name] + rows, open(out, "w"), indent=2)
