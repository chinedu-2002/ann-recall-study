"""Diagnose why our exact ground truth disagrees with the published ground truth
on some datasets. Hypothesis: exact distance ties, where two different vectors
sit at identical distance from the query and either ordering is correct.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np, datasets, groundtruth

out = {}
for name in ["sift-128-euclidean", "nytimes-256-angular"]:
    base, q, metric, published = datasets.load(name, n_base=None, n_query=1000)
    ours, _ = groundtruth.exact_knn(base, q, 10, metric)

    def dist(a, b):
        if metric == "euclidean":
            return np.linalg.norm(a - b, axis=-1)
        return 1.0 - (a * b).sum(axis=-1)

    mismatch = 0; tie_explained = 0; maxgap = 0.0
    for i in range(q.shape[0]):
        if ours[i, 0] == published[i, 0]:
            continue
        mismatch += 1
        d_ours = dist(q[i], base[ours[i, 0]])
        d_pub = dist(q[i], base[published[i, 0]])
        gap = abs(float(d_ours) - float(d_pub))
        maxgap = max(maxgap, gap)
        if gap <= 1e-5 * max(1.0, abs(float(d_pub))):
            tie_explained += 1
    out[name] = {"queries": int(q.shape[0]), "top1_id_mismatches": mismatch,
                 "explained_by_exact_distance_tie": tie_explained,
                 "max_distance_gap_among_mismatches": maxgap}
    print(name, json.dumps(out[name]))

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tie_investigation.json"), "w") as f:
    json.dump(out, f, indent=2)
