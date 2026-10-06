"""Parameter sweep driver.

For each (dataset, index family) pair it builds every index in the construction
grid once, then sweeps the query-time parameter over that built index, recording
recall, latency, build time and memory for every configuration. Results stream
to CSV so a long run can be inspected while it is still going.

Measurement decisions that matter:

1. Recall@k is measured from a result set of exactly k, because hnswlib silently
   raises ef to at least k. Sweeping ef below k while asking for k neighbors
   produces identical rows and looks like a flat region that does not exist.
2. Recall is reported both identifier-wise and tie-aware (distance-wise), since
   duplicate vectors make identifier recall understate a correct result.
3. The spread of per-query recall is recorded, so every recall figure can carry
   a 95% confidence interval over queries (mean +/- 1.96 * sd / sqrt(n)).
4. Search is always single-threaded and timed per query. Index construction may
   use more threads (ANN_BUILD_THREADS); the thread count is recorded per row.

Profiles select the grid:
    pr1    the Progress Report 1 grid (three families, coarse query sweep), k=10
    dense  default HNSW build and IVF-Flat nlist=1024 with a fine query sweep,
           used to find the cheapest setting that reaches a target recall
    kpass  default HNSW and IVF builds swept at k=1 and k=100
    scale  full-size base sets, default HNSW plus IVF nlist 100 and 4096, k=10
"""
import os, sys, time, csv, json, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets, groundtruth, indexes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

FIELDS = ["dataset", "n_base", "dim", "metric", "family", "build_params", "query_param", "k",
          "is_default", "recall_id", "recall_tieaware", "recall_q_std", "n_query",
          "latency_median_ms", "latency_p95_ms", "build_seconds", "build_threads",
          "memory_mb", "repeat"]

# ---- Progress Report 1 grid -------------------------------------------------
HNSW_BUILD = [{"M": 8, "efConstruction": 100}, {"M": 16, "efConstruction": 200},
              {"M": 32, "efConstruction": 200}, {"M": 48, "efConstruction": 400}]
IVF_BUILD = [{"nlist": 100}, {"nlist": 1024}, {"nlist": 4096}]
# Product quantization requires the subquantizer count m to divide the vector
# dimension exactly, so the grid cannot be shared verbatim across datasets of
# different width. We request a target m and snap it down to the nearest divisor.
PQ_TARGETS = [{"nlist": 1024, "m": 8, "nbits": 8}, {"nlist": 1024, "m": 16, "nbits": 8},
              {"nlist": 4096, "m": 16, "nbits": 8}]
HNSW_EF = [10, 16, 32, 64, 128, 256, 512]
IVF_NPROBE = [1, 2, 4, 8, 16, 32, 64, 128]
PQ_NPROBE = [1, 2, 4, 8, 16, 64, 128]

HNSW_DEFAULT = {"M": 16, "efConstruction": 200}
HNSW_DEFAULT_EF = 10
IVF_DEFAULT = {"nlist": 100}
IVF_DEFAULT_NPROBE = 1
PQ_DEFAULT = {"nlist": 1024, "m": 8, "nbits": 8}

# ---- dense grids for the heuristic ------------------------------------------
DENSE_EF = [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256,
            320, 384, 512, 640, 768, 1024]
DENSE_NPROBE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

# ---- k passes ----------------------------------------------------------------
# ef = 10 is kept in the k = 100 grid on purpose: hnswlib raises it to 100
# internally, so that row should match the ef = 100 row exactly. It is the
# direct evidence for the clamp, and it is what the library default does at k = 100.
KPASS_EF = {1: [1, 2, 4, 8, 10, 16, 32, 64, 128],
            100: [10, 100, 128, 192, 256, 384, 512, 768, 1024]}


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


def plan(profile, family, dim):
    """Return (build_grid, [(k, query_grid), ...], default_build, default_qp)."""
    if profile == "pr1":
        if family == "HNSW":
            return HNSW_BUILD, [(10, HNSW_EF)], HNSW_DEFAULT, HNSW_DEFAULT_EF
        if family == "IVF-Flat":
            return IVF_BUILD, [(10, IVF_NPROBE)], IVF_DEFAULT, IVF_DEFAULT_NPROBE
        if family == "IVF-PQ":
            d = dict(PQ_DEFAULT, m=_snap_m(PQ_DEFAULT["m"], dim))
            return pq_build_grid(dim), [(10, PQ_NPROBE)], d, 1
    if profile == "dense":
        if family == "HNSW":
            return [HNSW_DEFAULT], [(10, DENSE_EF)], HNSW_DEFAULT, HNSW_DEFAULT_EF
        if family == "IVF-Flat":
            return [{"nlist": 1024}], [(10, DENSE_NPROBE)], IVF_DEFAULT, IVF_DEFAULT_NPROBE
    if profile == "kpass":
        if family == "HNSW":
            return [HNSW_DEFAULT], [(k, g) for k, g in KPASS_EF.items()], HNSW_DEFAULT, HNSW_DEFAULT_EF
        if family == "IVF-Flat":
            return ([IVF_DEFAULT, {"nlist": 1024}], [(1, IVF_NPROBE), (100, IVF_NPROBE)],
                    IVF_DEFAULT, IVF_DEFAULT_NPROBE)
    if profile == "scale":
        if family == "HNSW":
            return [HNSW_DEFAULT], [(10, HNSW_EF)], HNSW_DEFAULT, HNSW_DEFAULT_EF
        if family == "IVF-Flat":
            return [IVF_DEFAULT, {"nlist": 4096}], [(10, IVF_NPROBE)], IVF_DEFAULT, IVF_DEFAULT_NPROBE
    raise ValueError(f"no plan for profile={profile} family={family}")


def dists_to(queries, base, ids, metric):
    """Exact distance from each query to each returned identifier."""
    v = base[np.clip(ids, 0, base.shape[0] - 1)]
    if metric == "euclidean":
        d = np.linalg.norm(queries[:, None, :] - v, axis=2)
    else:
        d = 1.0 - np.einsum("ij,ikj->ik", queries, v)
    return np.where(ids < 0, np.inf, d)


def per_query_tieaware(found_d, truth_d, k, rtol=1e-5):
    thresh = truth_d[:, k - 1:k]
    tol = rtol * np.maximum(1.0, np.abs(thresh))
    return (found_d[:, :k] <= thresh + tol).sum(axis=1) / k


def run_family(name, base, q, metric, truth_ids, truth_d, family, profile, writer, fh, repeats):
    dim = base.shape[1]
    cls = {"HNSW": indexes.HNSWIndex, "IVF-Flat": indexes.IVFFlatIndex,
           "IVF-PQ": indexes.IVFPQIndex}[family]
    build_grid, passes, dflt_build, dflt_qp = plan(profile, family, dim)
    for bp in build_grid:
        ix = cls(dim, metric, **bp)
        bt = ix.build(base)
        mem = ix.memory_bytes() / 1e6
        for k, query_grid in passes:
            for qp in query_grid:
                ix.set_query_param(qp)
                ix.search(q[:20], k)                          # warm up
                for r in range(repeats):
                    found, times = ix.search(q, k)
                    t = np.asarray(times) * 1000.0
                    fd = dists_to(q, base, found, metric)
                    pq = per_query_tieaware(fd, truth_d, k)
                    row = {"dataset": name, "n_base": base.shape[0], "dim": dim, "metric": metric,
                           "family": family, "build_params": json.dumps(bp, sort_keys=True),
                           "query_param": qp, "k": k,
                           "is_default": int(bp == dflt_build and qp == dflt_qp),
                           "recall_id": round(groundtruth.recall_at_k(found, truth_ids, k), 5),
                           "recall_tieaware": round(float(pq.mean()), 5),
                           "recall_q_std": round(float(pq.std(ddof=1)), 5),
                           "n_query": int(q.shape[0]),
                           "latency_median_ms": round(float(np.median(t)), 4),
                           "latency_p95_ms": round(float(np.percentile(t, 95)), 4),
                           "build_seconds": round(bt, 2), "build_threads": indexes.BUILD_THREADS,
                           "memory_mb": round(mem, 2), "repeat": r}
                    writer.writerow(row); fh.flush()
                print(f"{name} {family} {bp} k={k} qp={qp} "
                      f"rec={row['recall_tieaware']:.4f} med={row['latency_median_ms']:.3f}ms",
                      flush=True)
        del ix; gc.collect()


def main(dataset, n_base, n_query, families, profile, out_csv, repeats):
    base, q, metric, _ = datasets.load(dataset, n_base=n_base, n_query=n_query)
    print("loaded %s: base=%s queries=%s metric=%s profile=%s" %
          (dataset, base.shape, q.shape, metric, profile), flush=True)
    kmax = 100 if profile == "kpass" else 10
    t0 = time.perf_counter()
    truth_ids, raw = groundtruth.exact_knn(base, q, kmax, metric)
    truth_d = (1.0 - raw) if metric == "angular" else np.sqrt(np.maximum(raw, 0))
    print("exact ground truth in %.1fs" % (time.perf_counter() - t0), flush=True)

    exists = os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if not exists:
        w.writeheader()
    for fam in families:
        run_family(dataset, base, q, metric, truth_ids, truth_d, fam, profile, w, fh, repeats)
    fh.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-base", type=int, default=200000)
    p.add_argument("--n-query", type=int, default=300)
    p.add_argument("--profile", default="pr1", choices=["pr1", "dense", "kpass", "scale"])
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    fams = a.families or (["HNSW", "IVF-Flat", "IVF-PQ"] if a.profile == "pr1" else ["HNSW", "IVF-Flat"])
    tag = "full" if a.n_base <= 0 else str(a.n_base)
    nb = None if a.n_base <= 0 else a.n_base
    suffix = "_k10" if a.profile == "pr1" else f"_{a.profile}"
    out = a.out or os.path.join(RESULTS, f"sweep_{a.dataset}_{tag}{suffix}.csv")
    main(a.dataset, nb, a.n_query, fams, a.profile, out, a.repeats)
