"""Live demo of the parameter-selection heuristic on a dataset it has never seen.

1. Generate a fresh synthetic dataset whose intrinsic dimension (20) and random
   seed were never used in fitting.
2. Measure its local intrinsic dimensionality (LID) from a sample, in seconds.
3. Predict the IVF nprobe needed for recall@10 >= 0.95 from the fitted model,
   plus the pre-registered one-residual safety margin.
4. Build the index and check the prediction against the library default and
   against the true cheapest setting found by sweeping.

Needs only numpy and faiss. Coefficients come from results/pr2_report_numbers.json.
    python scripts/predict_demo.py
"""
import os, json, time
import numpy as np
import faiss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET, K, NLIST, N, NQ, DIM, ID_TRUE = 0.95, 10, 1024, 200_000, 300, 128, 20

fit = json.load(open(os.path.join(ROOT, "results", "pr2_report_numbers.json")))["fit_IVF-Flat"]
a, b, sigma = fit["a"], fit["b"], fit["sigma_log2"]

line = "=" * 64
print(f"\n{line}\n  Predicting an index setting from the data itself\n{line}")

# 1. a dataset the model never saw: same generator family, new dimension and seed
t0 = time.perf_counter()
rng = np.random.default_rng(777)
centers = rng.normal(scale=2.0, size=(20, ID_TRUE))
W1 = rng.normal(scale=1 / np.sqrt(ID_TRUE), size=(ID_TRUE, 256)); b1 = rng.normal(scale=0.1, size=256)
W2 = rng.normal(scale=1 / np.sqrt(256), size=(256, DIM))
def sample(n):
    z = centers[rng.integers(0, 20, n)] + rng.normal(size=(n, ID_TRUE))
    return (np.tanh(z @ W1 + b1) @ W2 + rng.normal(scale=0.02, size=(n, DIM))).astype("float32")
base, q = sample(N), sample(NQ)
print(f"\n1. fresh dataset: {N:,} vectors x {DIM} dims  ({time.perf_counter()-t0:.1f}s)")

# 2. LID at the query points, estimated against a 20,000-vector sample
t0 = time.perf_counter()
samp = base[rng.choice(N, 20_000, replace=False)]
ix = faiss.IndexFlatL2(DIM); ix.add(samp)
d, _ = ix.search(q, 21)
d = np.sqrt(np.maximum(np.sort(d, axis=1)[:, 1:], 1e-12))
lid = -1.0 / np.mean(np.log(d[:, :-1] / d[:, -1:]), axis=1)
lid = lid[np.isfinite(lid) & (lid > 0)]
lo, hi = np.percentile(lid, [5, 95]); lid = float(np.mean(lid[(lid >= lo) & (lid <= hi)]))
print(f"2. measured LID at the queries: {lid:.1f}  ({time.perf_counter()-t0:.1f}s, no index built)")

# 3. predict
raw = 2 ** (a + b * lid)
pred = int(np.ceil(raw * 2 ** sigma))
print(f"3. model: log2(nprobe) = {a:.2f} + {b:.3f} x LID")
print(f"   prediction {raw:.1f}, with safety margin -> nprobe = {pred}")

# 4. check it
flat = faiss.IndexFlatL2(DIM); flat.add(base)
_, truth = flat.search(q, K)
ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(DIM), DIM, NLIST, faiss.METRIC_L2)
t0 = time.perf_counter(); ivf.train(base); ivf.add(base)
print(f"\n4. built IVF index, nlist={NLIST}  ({time.perf_counter()-t0:.1f}s)")
faiss.omp_set_num_threads(1)

def measure(nprobe):
    ivf.nprobe = nprobe
    ivf.search(q[:20], K)
    t = time.perf_counter(); _, ids = ivf.search(q, K); el = (time.perf_counter() - t) / NQ * 1000
    rec = np.mean([len(set(ids[i]) & set(truth[i])) / K for i in range(NQ)])
    return rec, el

rd, ld = measure(1)
rp, lp = measure(pred)
oracle = None
for npb in range(1, 257):
    r, l = measure(npb)
    if r >= TARGET:
        oracle, ro, lo_ = npb, r, l
        break

print(f"\n   {'setting':<26}{'nprobe':>7}{'recall@10':>11}{'ms/query':>10}")
print(f"   {'library default':<26}{1:>7}{rd:>11.3f}{ld:>10.3f}")
print(f"   {'heuristic prediction':<26}{pred:>7}{rp:>11.3f}{lp:>10.3f}")
if oracle:
    print(f"   {'true cheapest (swept)':<26}{oracle:>7}{ro:>11.3f}{lo_:>10.3f}")
print(f"\n   target {TARGET}: prediction {'MET' if rp >= TARGET else 'MISSED'}", end="")
if oracle:
    print(f", at {lp / lo_:.2f}x the cost of the true cheapest setting")
print(f"{line}\n")
