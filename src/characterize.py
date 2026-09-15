"""Measurable dataset properties hypothesized to predict good index parameters.

These are the candidate features for the parameter-selection heuristic that is
the research contribution of the project. All of them are computable from a
sample of the base set without building any index.
"""
import numpy as np
import faiss
from sklearn.cluster import KMeans


def _knn_distances(sample, reference, k, metric):
    d = reference.shape[1]
    index = faiss.IndexFlatL2(d) if metric == "euclidean" else faiss.IndexFlatIP(d)
    index.add(reference)
    dist, _ = index.search(sample, k + 1)
    if metric == "angular":
        # convert inner product of unit vectors to a true distance
        dist = np.sqrt(np.maximum(2.0 - 2.0 * dist, 0.0))
    else:
        dist = np.sqrt(np.maximum(dist, 0.0))
    return np.sort(dist, axis=1)[:, 1:]  # drop self-match


def intrinsic_dimension_mle(sample, reference, metric, k=20, min_valid=10):
    """Levina-Bickel maximum-likelihood intrinsic dimension estimator.

    A dataset whose points lie on a low-dimensional manifold inside a
    high-dimensional ambient space reports a value well below its column count.

    Exact duplicate vectors are the failure mode here: a neighbor at distance
    zero drives the per-point estimate to infinity, and on a corpus where even
    two percent of vectors are duplicated the plain sample mean is dominated by
    those points. We therefore drop zero distances, require a minimum number of
    usable neighbors per point, and aggregate with a trimmed mean.
    """
    dists = _knn_distances(sample, reference, k, metric)
    est = []
    for row in dists:
        pos = row[row > 1e-9]
        if pos.size < min_valid:
            continue
        logs = np.log(pos)
        inv = (logs[-1] - logs[:-1]).mean()
        if inv > 1e-9:
            est.append(1.0 / inv)
    est = np.asarray(est)
    est = est[np.isfinite(est)]
    if est.size == 0:
        return float("nan")
    lo, hi = np.percentile(est, [5, 95])
    trimmed = est[(est >= lo) & (est <= hi)]
    return float(np.mean(trimmed))


def local_intrinsic_dimension(queries, reference, metric, k=20):
    """Mean and spread of LID measured at the actual query points.

    High LID around queries is the regime where graph indexes are known to
    struggle, so this is the property most likely to carry predictive signal.
    """
    dists = _knn_distances(queries, reference, k, metric)
    dists = np.maximum(dists, 1e-12)
    w = dists[:, -1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lid = -1.0 / np.mean(np.log(dists[:, :-1] / w), axis=1)
    lid = lid[np.isfinite(lid) & (lid > 0)]
    lo, hi = np.percentile(lid, [5, 95])
    lid = lid[(lid >= lo) & (lid <= hi)]
    return float(np.mean(lid)), float(np.std(lid))


def cluster_structure(sample, k_values=(8, 16, 32, 64, 128), seed=0):
    """Relative inertia drop across k-means settings.

    A dataset with strong cluster structure shows a steep inertia curve, which
    should favor inverted-file indexes with few probes. A dataset with diffuse
    structure shows a shallow curve.
    """
    # MiniBatchKMeans inertia is not reliably monotonic in k on high-dimensional
    # normalized data, which produced a positive (nonsensical) slope on the text
    # corpus. Full Lloyd's algorithm with multiple restarts is slower but stable.
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=100)
        km.fit(sample)
        inertias.append(km.inertia_)
    inertias = np.asarray(inertias, dtype=float)
    # slope of log(inertia) vs log(k): steeper (more negative) = stronger clustering
    slope = np.polyfit(np.log(k_values), np.log(inertias), 1)[0]
    return {"inertia_log_slope": float(slope),
            "inertia_ratio_8_to_128": float(inertias[0] / inertias[-1])}


def characterize(base, queries, metric, n_sample=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n_sample, base.shape[0])
    idx = rng.choice(base.shape[0], size=n, replace=False)
    sample = np.ascontiguousarray(base[idx])
    qs = np.ascontiguousarray(queries[:min(1000, queries.shape[0])])

    lid_mean, lid_std = local_intrinsic_dimension(qs, sample, metric)
    out = {
        "n_base": int(base.shape[0]),
        "ambient_dim": int(base.shape[1]),
        "metric": metric,
        "intrinsic_dim_mle": intrinsic_dimension_mle(sample, sample, metric),
        "lid_at_queries_mean": lid_mean,
        "lid_at_queries_std": lid_std,
    }
    out.update(cluster_structure(sample, seed=seed))
    out["dim_ratio"] = out["intrinsic_dim_mle"] / out["ambient_dim"]
    return out
