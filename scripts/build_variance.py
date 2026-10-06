"""How much does recall at the default setting change when the *same* index is
rebuilt? hnswlib inserts in parallel when given more than one thread, so the
graph depends on thread scheduling. This rebuilds the default HNSW configuration
several times per thread count and measures recall@10 at ef = 10 each time."""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, hnswlib
import datasets, groundtruth
from sweep import dists_to, per_query_tieaware

rows = []
NAMES = sys.argv[1:] or ["sift-128-euclidean", "nytimes-256-angular", "glove-100-angular"]
for name in NAMES:
    base, q, metric, _ = datasets.load(name, n_base=200000, n_query=300)
    tids, raw = groundtruth.exact_knn(base, q, 10, metric)
    td = (1.0 - raw) if metric == "angular" else np.sqrt(np.maximum(raw, 0))
    for threads, reps in ((1, 2), (2, 4)):
        for r in range(reps):
            ix = hnswlib.Index(space="l2" if metric == "euclidean" else "ip", dim=base.shape[1])
            ix.init_index(max_elements=base.shape[0], ef_construction=200, M=16, random_seed=42)
            ix.set_num_threads(threads)
            t0 = time.perf_counter(); ix.add_items(base, np.arange(base.shape[0])); bt = time.perf_counter() - t0
            ix.set_ef(10); ix.set_num_threads(1)
            ids, _ = ix.knn_query(q, k=10)
            ids = ids.astype(np.int64)
            rec_id = groundtruth.recall_at_k(ids, tids, 10)
            rec_ta = float(per_query_tieaware(dists_to(q, base, ids, metric), td, 10).mean())
            rows.append({"dataset": name, "build_threads": threads, "rebuild": r,
                         "recall_tieaware_ef10": round(rec_ta, 5), "recall_id_ef10": round(rec_id, 5),
                         "build_s": round(bt, 1)})
            print(json.dumps(rows[-1]), flush=True)
out = os.path.join(ROOT, "results", "build_variance.json")
prev = json.load(open(out)) if os.path.exists(out) else []
prev = [r for r in prev if r["dataset"] not in NAMES]
json.dump(prev + rows, open(out, "w"), indent=2)
