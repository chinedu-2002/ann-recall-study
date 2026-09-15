"""Re-run the ground-truth validation using tie-aware (distance-based) recall.

Identifier agreement understates correctness on datasets containing duplicate
vectors. This script shows the same comparison measured on distances.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np, datasets, groundtruth

rows = []
for name in ["sift-128-euclidean", "glove-100-angular", "nytimes-256-angular"]:
    base, q, metric, published = datasets.load(name, n_base=None, n_query=1000)
    ours_ids, ours_d = groundtruth.exact_knn(base, q, 100, metric)
    if metric == "angular":
        ours_d = 1.0 - ours_d           # inner product -> cosine distance
        pub_d = 1.0 - np.einsum("ij,ikj->ik", q, base[published])
    else:
        ours_d = np.sqrt(np.maximum(ours_d, 0))
        pub_d = np.linalg.norm(q[:, None, :] - base[published], axis=2)
    row = {"dataset": name}
    for k in (1, 10, 100):
        row[f"id_agreement@{k}"] = round(groundtruth.recall_at_k(ours_ids, published, k), 5)
        row[f"tieaware_agreement@{k}"] = round(
            groundtruth.recall_at_k_tie_aware(ours_d, pub_d, k), 5)
    print(json.dumps(row), flush=True)
    rows.append(row)

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                 "groundtruth_validation_tieaware.json")
json.dump(rows, open(p, "w"), indent=2)
