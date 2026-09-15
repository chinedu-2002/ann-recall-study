"""Parameter sweep driver.

For each (dataset, index family) pair it builds every index in the construction
grid once, then sweeps the query-time parameter over that built index, recording
recall, latency, build time and memory for every configuration. Results stream
to CSV so a long run can be inspected while it is still going.

Two measurement decisions matter here:

1. Recall@k is measured from a result set of exactly k, because hnswlib silently
   raises ef to at least k. Sweeping ef below k while asking for k neighbors
   produces identical rows and looks like a flat region that does not exist.
2. Recall is reported both identifier-wise and tie-aware (distance-wise), since
   duplicate vectors make identifier recall understate a correct result.
"""
import os, sys, time, csv, json, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets, groundtruth, indexes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

FIELDS = ["dataset", "n_base", "dim", "metric", "family", "build_params", "query_param", "k",
          "is_default", "recall_id", "recall_tieaware",
          "latency_median_ms", "latency_p95_ms", "build_seconds", "memory_mb", "repeat"]

HNSW_BUILD = [{"M": 8, "efConstruction": 100}, {"M": 16, "efConstruction": 200},
              {"M": 32, "efConstruction": 200}, {"M": 48, "efConstruction": 400}]
IVF_BUILD = [{"nlist": 100}, {"nlist": 1024}, {"nlist": 4096}]
# Product quantization requires the subquantizer count m to divide the vector
# dimension exactly, so the grid cannot be shared verbatim across datasets of
# different width. We request a target m and snap it down to the nearest divisor.
PQ_TARGETS = [{"nlist": 1024, "m": 8, "nbits": 8}, {"nlist": 1024, "m": 16, "nbits": 8},
              {"nlist": 4096, "m": 16, "nbits": 8}]


def _snap_m(target, dim):
    """Largest divisor of dim that does not exceed target (falls back upward)."""
    div = [x for x in range(1, dim + 1) if dim % x == 0]
    below = [x for x in div if x <= target]
    return max(below) if below else min(div)


def pq_build_grid(dim):
    seen, grid = set(), []
    for t in PQ_TARGETS:
        g = dict(t, m=_snap_m(t["m"], dim))
        key = tuple(sorted(g.items()))
        if key not in seen:
            seen.add(key)
            grid.append(g)
    return grid

HNSW_EF = [10, 16, 32, 64, 128, 256, 512]
IVF_NPROBE = [1, 2, 4, 8, 16, 32, 64, 128]
PQ_NPROBE = [1, 2, 4, 8, 16, 64, 128]

GRIDS = {
    "HNSW":     (HNSW_BUILD, HNSW_EF,    ({"M": 16, "efConstruction": 200}, 10)),
    "IVF-Flat": (IVF_BUILD,  IVF_NPROBE, ({"nlist": 100}, 1)),
    "IVF-PQ":   (None,       PQ_NPROBE,  ({"nlist": 1024, "m": 8, "nbits": 8}, 1)),
}


def dists_to(queries, base, ids, metric):
    """Exact distance from each query to each returned identifier."""
    v = base[np.clip(ids, 0, base.shape[0] - 1)]
    if metric == "euclidean":
        d = np.linalg.norm(queries[:, None, :] - v, axis=2)
    else:
        d = 1.0 - np.einsum("ij,ikj->ik", queries, v)
    d = np.where(ids < 0, np.inf, d)
    return d


def run_family(name, base, q, metric, truth_ids, truth_d, k, family, writer, fh, repeats):
    dim = base.shape[1]
    cls = {"HNSW": indexes.HNSWIndex, "IVF-Flat": indexes.IVFFlatIndex,
           "IVF-PQ": indexes.IVFPQIndex}[family]
    build_grid, query_grid, dflt = GRIDS[family]
    if family == "IVF-PQ":
        build_grid = pq_build_grid(dim)
        dflt = (dict(dflt[0], m=_snap_m(dflt[0]["m"], dim)), dflt[1])
    for bp in build_grid:
        ix = cls(dim, metric, **bp)
        bt = ix.build(base)
        mem = ix.memory_bytes() / 1e6
        for qp in query_grid:
            ix.set_query_param(qp)
            ix.search(q[:20], k)
            for r in range(repeats):
                found, times = ix.search(q, k)
                t = np.asarray(times) * 1000.0
                fd = dists_to(q, base, found, metric)
                row = {"dataset": name, "n_base": base.shape[0], "dim": dim, "metric": metric,
                       "family": family, "build_params": json.dumps(bp, sort_keys=True),
                       "query_param": qp, "k": k,
                       "is_default": int(bp == dflt[0] and qp == dflt[1]),
                       "recall_id": round(groundtruth.recall_at_k(found, truth_ids, k), 5),
                       "recall_tieaware": round(
                           groundtruth.recall_at_k_tie_aware(fd, truth_d, k), 5),
                       "latency_median_ms": round(float(np.median(t)), 4),
                       "latency_p95_ms": round(float(np.percentile(t, 95)), 4),
                       "build_seconds": round(bt, 2), "memory_mb": round(mem, 2), "repeat": r}
                writer.writerow(row); fh.flush()
                print(f"{name} {family} {bp} qp={qp} r{r} "
                      f"rec={row['recall_tieaware']:.4f} med={row['latency_median_ms']:.3f}ms",
                      flush=True)
        del ix; gc.collect()


def main(dataset, n_base, n_query, families, k, out_csv, repeats):
    base, q, metric, _ = datasets.load(dataset, n_base=n_base, n_query=n_query)
    print("loaded %s: base=%s queries=%s metric=%s" % (dataset, base.shape, q.shape, metric), flush=True)
    t0 = time.perf_counter()
    truth_ids, raw = groundtruth.exact_knn(base, q, k, metric)
    truth_d = (1.0 - raw) if metric == "angular" else np.sqrt(np.maximum(raw, 0))
    print("exact ground truth in %.1fs" % (time.perf_counter() - t0), flush=True)

    exists = os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if not exists:
        w.writeheader()
    for fam in families:
        run_family(dataset, base, q, metric, truth_ids, truth_d, k, fam, w, fh, repeats)
    fh.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-base", type=int, default=200000)
    p.add_argument("--n-query", type=int, default=300)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--families", nargs="+", default=["HNSW", "IVF-Flat", "IVF-PQ"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(RESULTS, "sweep_%s_%d_k%d.csv" % (a.dataset, a.n_base, a.k))
    main(a.dataset, a.n_base, a.n_query, a.families, a.k, out, a.repeats)
