"""Exact k-nearest-neighbor ground truth by brute force.

Recall is only meaningful against exact neighbors, so this module is the
foundation of every measurement in the study. It is deliberately simple:
exhaustive distance computation, no approximation anywhere.
"""
import numpy as np
import faiss


def exact_knn(base, queries, k, metric):
    """Exhaustive search. Returns (neighbor_ids, distances)."""
    d = base.shape[1]
    if metric == "euclidean":
        index = faiss.IndexFlatL2(d)
    elif metric == "angular":
        # vectors are L2-normalized upstream, so maximum inner product
        # ranks identically to minimum cosine distance
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError(metric)
    index.add(base)
    dist, ids = index.search(queries, k)
    return ids.astype(np.int32), dist


def recall_at_k(found, truth, k):
    """Mean fraction of the true k nearest neighbors present in the result set."""
    hits = 0
    n = truth.shape[0]
    for i in range(n):
        hits += len(set(found[i, :k].tolist()) & set(truth[i, :k].tolist()))
    return hits / (n * k)


def recall_at_k_tie_aware(found_dists, truth_dists, k, rtol=1e-5):
    """Recall measured on distances rather than identifiers.

    When two base vectors sit at exactly the same distance from a query, either
    is a correct nearest neighbor, but identifier-based recall arbitrarily
    penalizes whichever one the index happened to return. This counts a result
    as a hit when its distance matches the k-th true distance or better, which
    is the definition that survives duplicate vectors.
    """
    n = truth_dists.shape[0]
    hits = 0
    for i in range(n):
        thresh = truth_dists[i, k - 1]
        tol = rtol * max(1.0, abs(float(thresh)))
        hits += int(np.sum(found_dists[i, :k] <= thresh + tol))
    return hits / (n * k)
