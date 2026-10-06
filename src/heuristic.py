"""Parameter-selection heuristic: predict the query-time setting a dataset needs.

The question this module answers is whether a cheap statistic of a dataset can
stand in for an exhaustive parameter sweep. For a fixed index build and a target
recall, each dataset has an oracle setting: the smallest query parameter (HNSW ef,
or IVF nprobe) that reaches the target. The heuristic is a one-feature log-linear
model

    log2(setting) = a + b * feature

fitted across datasets. It is judged by leave-one-dataset-out evaluation on the
real corpora: fit on everything except one real dataset, predict that dataset's
setting, snap the prediction up to the next measured grid value, and read off the
recall and latency that setting actually produced. Two baselines keep it honest:

    constant-max     one setting for every dataset, the largest oracle value seen
                     in training ("pick a number that worked everywhere I tested")
    constant-median  the median training oracle value (cheaper, riskier)

The heuristic+margin variant inflates the prediction by one RMS training residual
(in log2 space), a margin fixed in advance rather than tuned on held-out data.

plus the library default and the per-dataset oracle.
"""
import numpy as np
import pandas as pd

FEATURES = {
    "lid": "lid_at_queries_mean",
    "intrinsic_dim": "intrinsic_dim_mle",
    "ambient_dim": "ambient_dim",
}


def aggregate(df):
    """Mean over timing repeats for each (dataset, family, build, qp, k)."""
    return (df.groupby(["dataset", "family", "build_params", "query_param", "k"], as_index=False)
              .agg(recall=("recall_tieaware", "mean"),
                   recall_q_std=("recall_q_std", "mean"),
                   n_query=("n_query", "first"),
                   latency_ms=("latency_median_ms", "mean"),
                   latency_sd=("latency_median_ms", "std"),
                   is_default=("is_default", "max")))


def oracle_table(agg, target):
    """Smallest measured setting reaching the target, per dataset and family."""
    rows = []
    for (ds, fam), g in agg.groupby(["dataset", "family"]):
        g = g.sort_values("query_param")
        ok = g[g["recall"] >= target]
        rows.append({"dataset": ds, "family": fam, "target": target,
                     "oracle_qp": float(ok["query_param"].iloc[0]) if len(ok) else np.nan,
                     "oracle_latency_ms": float(ok["latency_ms"].iloc[0]) if len(ok) else np.nan,
                     "max_recall": float(g["recall"].max())})
    return pd.DataFrame(rows)


def fit(x, y):
    """Least squares on log2(setting). Returns (a, b); b = 0 when x is None."""
    y = np.log2(np.asarray(y, dtype=float))
    if x is None:
        return float(y.mean()), 0.0
    b, a = np.polyfit(np.asarray(x, dtype=float), y, 1)
    return float(a), float(b)


def snap_up(value, grid):
    grid = np.sort(np.asarray(grid, dtype=float))
    above = grid[grid >= value - 1e-9]
    return float(above[0]) if len(above) else float(grid[-1])


def lookup(agg, ds, fam, qp):
    r = agg[(agg["dataset"] == ds) & (agg["family"] == fam) & (agg["query_param"] == qp)]
    return float(r["recall"].iloc[0]), float(r["latency_ms"].iloc[0])


def evaluate(agg, chars, family, target, feature, heldout_pool, train_pool, default_qp):
    """Leave-one-dataset-out over heldout_pool; training uses train_pool minus the held-out set."""
    orc = oracle_table(agg[agg["family"] == family], target).set_index("dataset")
    feat = chars.set_index("dataset")[FEATURES[feature]] if feature else None
    grid = sorted(agg[agg["family"] == family]["query_param"].unique())
    out = []
    for ds in heldout_pool:
        if ds not in orc.index or np.isnan(orc.loc[ds, "oracle_qp"]):
            continue
        train = [t for t in train_pool if t != ds and t in orc.index
                 and not np.isnan(orc.loc[t, "oracle_qp"])]
        ytr = orc.loc[train, "oracle_qp"].values
        xtr = feat.loc[train].values if feature else None
        a, b = fit(xtr, ytr)
        pred = 2 ** (a + b * (float(feat.loc[ds]) if feature else 0.0))
        # safety margin: one RMS training residual in log2 space, fixed in advance
        # rather than tuned on held-out data
        fitted = a + b * (np.asarray(xtr, dtype=float) if feature else 0.0)
        sigma = float(np.sqrt(np.mean((np.log2(ytr) - fitted) ** 2)))
        methods = {
            "heuristic": snap_up(pred, grid),
            "heuristic+margin": snap_up(pred * 2 ** sigma, grid),
            "constant-max": snap_up(float(np.max(ytr)), grid),
            "constant-median": snap_up(float(np.median(ytr)), grid),
            "default": float(default_qp),
            "oracle": float(orc.loc[ds, "oracle_qp"]),
        }
        o_rec, o_lat = lookup(agg, ds, family, methods["oracle"])
        for m, qp in methods.items():
            rec, lat = lookup(agg, ds, family, qp) if qp in grid else (np.nan, np.nan)
            out.append({"dataset": ds, "family": family, "target": target, "feature": feature,
                        "method": m, "setting": qp, "raw_prediction": pred if m == "heuristic" else np.nan,
                        "recall": rec, "latency_ms": lat, "hit": bool(rec >= target),
                        "latency_vs_oracle": lat / o_lat,
                        "log2_error": np.log2(qp) - np.log2(methods["oracle"])})
    return pd.DataFrame(out)


def summarize(ev):
    s = (ev.groupby(["family", "target", "feature", "method"])
           .agg(n=("dataset", "count"),
                hit_rate=("hit", "mean"),
                mean_recall=("recall", "mean"),
                min_recall=("recall", "min"),
                geo_latency_vs_oracle=("latency_vs_oracle", lambda v: float(np.exp(np.mean(np.log(v))))),
                mean_abs_log2_error=("log2_error", lambda v: float(np.mean(np.abs(v)))))
           .reset_index())
    return s
