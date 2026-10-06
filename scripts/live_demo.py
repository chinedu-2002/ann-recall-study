"""Live demo: what library default settings actually cost in recall."""
import time
import numpy as np
import faiss

N, D, NQ, K = 50_000, 64, 200, 10
rng = np.random.default_rng(0)
print("\n" + "=" * 62)
print("  ANN recall demo: what do library defaults cost you?")
print("=" * 62)

centers = rng.normal(scale=3.0, size=(40, D)).astype("float32")
base = (centers[rng.integers(0, 40, N)] + rng.normal(size=(N, D))).astype("float32")
queries = (centers[rng.integers(0, 40, NQ)] + rng.normal(size=(NQ, D))).astype("float32")
print(f"\ndataset: {N:,} vectors x {D} dims, {NQ} queries, recall@{K}")

t0 = time.perf_counter()
flat = faiss.IndexFlatL2(D)
flat.add(base)
_, truth = flat.search(queries, K)
print(f"exact ground truth by brute force in {time.perf_counter()-t0:.2f}s")


def measure(nlist, nprobe, label):
    faiss.omp_set_num_threads(1)
    ix = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, nlist, faiss.METRIC_L2)
    t0 = time.perf_counter()
    ix.train(base)
    ix.add(base)
    build = time.perf_counter() - t0
    ix.nprobe = nprobe
    ix.search(queries[:20], K)
    found = np.empty((NQ, K), dtype=np.int64)
    times = []
    for i in range(NQ):
        t1 = time.perf_counter()
        _, ids = ix.search(queries[i:i+1], K)
        times.append(time.perf_counter() - t1)
        found[i] = ids[0]
    rec = sum(len(set(found[i]) & set(truth[i])) for i in range(NQ)) / (NQ * K)
    lat = float(np.median(times)) * 1000
    print(f"\n  {label}")
    print(f"    nlist={nlist}, nprobe={nprobe}   build {build:.2f}s")
    print(f"    recall@{K}        {rec:.3f}")
    print(f"    median latency   {lat:.3f} ms")
    return rec, lat


print("\n" + "-" * 62)
rd, ld = measure(100, 1, "FAISS DEFAULT SETTINGS")
rt, lt = measure(1024, 16, "TUNED SETTINGS")
print("\n" + "=" * 62)
print(f"  recall gained by tuning:  {rt-rd:+.3f}   ({rd:.1%} -> {rt:.1%})")
print(f"  latency change:           {lt-ld:+.3f} ms")
print(f"\n  The default silently returns {1-rd:.0%} wrong neighbors.")
print("  No error. No warning. The query just comes back worse.")
print("=" * 62 + "\n")
