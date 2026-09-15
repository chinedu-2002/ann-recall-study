"""Dataset loading for the ANN recall study.

All datasets are the standard ANN-Benchmarks HDF5 files, which ship with a
fixed train/test split and published ground-truth neighbors. We reload them,
optionally subsample the base set, and (for angular datasets) L2-normalize so
that inner product is equivalent to cosine similarity.
"""
import os
import numpy as np
import h5py

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASETS = {
    "sift-128-euclidean":  {"metric": "euclidean", "domain": "image descriptors (SIFT)"},
    "glove-100-angular":   {"metric": "angular",   "domain": "word embeddings (GloVe)"},
    "nytimes-256-angular": {"metric": "angular",   "domain": "text documents (NYTimes)"},
}


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def load(name, n_base=None, n_query=None, seed=0):
    """Return (base, queries, metric, published_neighbors_or_None).

    published_neighbors is only returned when the base set is NOT subsampled,
    because the shipped ground truth indexes into the full base set.
    """
    path = os.path.join(DATA_DIR, name + ".hdf5")
    with h5py.File(path, "r") as f:
        base = np.asarray(f["train"], dtype=np.float32)
        queries = np.asarray(f["test"], dtype=np.float32)
        published = np.asarray(f["neighbors"], dtype=np.int32)
    metric = DATASETS[name]["metric"]

    full_base = n_base is None or n_base >= base.shape[0]
    if not full_base:
        rng = np.random.default_rng(seed)
        idx = rng.choice(base.shape[0], size=n_base, replace=False)
        idx.sort()
        base = base[idx]
        published = None
    if n_query is not None and n_query < queries.shape[0]:
        queries = queries[:n_query]
        if published is not None:
            published = published[:n_query]

    if metric == "angular":
        base = _normalize(base)
        queries = _normalize(queries)
    return np.ascontiguousarray(base), np.ascontiguousarray(queries), metric, published
