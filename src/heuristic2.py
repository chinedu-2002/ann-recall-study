"""Week 7: multi-feature version of the parameter-selection model.

Same idea as heuristic.py, generalized from one feature to several:

    log2(setting) = a + b1 * LID + b2 * log2(N) + b3 * hubness

The size sweep gives SIFT and GloVe-100 at five collection sizes, so each
(dataset, size) pair is one training row. Leave-one-dataset-out removes every
size of the held-out dataset from training, so the model never sees any version
of the collection it is asked about.
"""
import numpy as np
import pandas as pd

MODELS = {
    "none": [],
    "LID": ["lid"],
    "LID + size": ["lid", "log2n"],
    "LID + hubness": ["lid", "hub"],
    "LID + size + hubness": ["lid", "log2n", "hub"],
}


def aggregate(df):
    return (df.groupby(["dataset", "n_base", "family", "query_param", "k"], as_index=False)
              .agg(recall=("recall_tieaware", "mean"), latency_ms=("latency_median_ms", "mean")))


def oracle_rows(agg, chars, target):
    c = chars.set_index("dataset")
    rows = []
    for (ds, n, fam), g in agg.groupby(["dataset", "n_base", "family"]):
        g = g.sort_values("query_param")
        ok = g[g["recall"] >= target]
        if not len(ok):
            continue
        rows.append({"dataset": ds, "n_base": int(n), "family": fam, "target": target,
                     "oracle_qp": float(ok["query_param"].iloc[0]),
                     "oracle_latency_ms": float(ok["latency_ms"].iloc[0]),
                     "lid": float(c.loc[ds, "lid_at_queries_mean"]),
                     "hub": float(c.loc[ds, "hubness_skew_k10"]),
                     "log2n": float(np.log2(n))})
    return pd.DataFrame(rows)


def fit(X, y):
    A = np.column_stack([np.ones(len(y))] + ([X] if X.size else []))
    coef, *_ = np.linalg.lstsq(A, np.log2(y), rcond=None)
    resid = np.log2(y) - A @ coef
    return coef, float(np.sqrt(np.mean(resid ** 2)))


def predict(coef, x):
    return float(2 ** (coef[0] + (np.dot(coef[1:], x) if len(coef) > 1 else 0.0)))


def snap_up(v, grid):
    g = np.sort(np.asarray(grid, float)); a = g[g >= v - 1e-9]
    return float(a[0]) if len(a) else float(g[-1])


def evaluate(agg, orc, family, model, heldout, test_sizes="primary"):
    """Leave one dataset out (all its sizes). test_sizes='primary' tests only the
    size used in weeks 5 and 6 (200,000, or the full set when smaller); 'all'
    tests every size available for the held-out dataset."""
    feats = MODELS[model]
    o = orc[orc["family"] == family]
    out = []
    for ds in heldout:
        test = o[o["dataset"] == ds]
        if not len(test):
            continue
        if test_sizes == "primary":
            primary = 200000 if (test["n_base"] == 200000).any() else int(test["n_base"].min())
            test = test[test["n_base"] == primary]
        train = o[o["dataset"] != ds]
        X = train[feats].values if feats else np.empty((len(train), 0))
        coef, sigma = fit(X, train["oracle_qp"].values)
        for _, t in test.iterrows():
            grid = sorted(agg[(agg["dataset"] == ds) & (agg["n_base"] == t["n_base"]) &
                              (agg["family"] == family)]["query_param"].unique())
            raw = predict(coef, t[feats].values.astype(float)) if feats else predict(coef, [])
            for m, qp in (("model", snap_up(raw, grid)), ("model + margin", snap_up(raw * 2 ** sigma, grid))):
                r = agg[(agg["dataset"] == ds) & (agg["n_base"] == t["n_base"]) & (agg["family"] == family)
                        & (agg["query_param"] == qp)].iloc[0]
                out.append({"dataset": ds, "n_base": int(t["n_base"]), "family": family, "model": model,
                            "method": m, "target": t["target"], "setting": qp, "oracle_qp": t["oracle_qp"],
                            "recall": float(r["recall"]), "hit": bool(r["recall"] >= t["target"]),
                            "latency_vs_oracle": float(r["latency_ms"]) / t["oracle_latency_ms"],
                            "abs_log2_error": abs(np.log2(qp) - np.log2(t["oracle_qp"]))})
    return pd.DataFrame(out)


def summarize(ev):
    return (ev.groupby(["family", "target", "model", "method"])
              .agg(n=("dataset", "count"), hits=("hit", "sum"), min_recall=("recall", "min"),
                   cost=("latency_vs_oracle", lambda v: float(np.exp(np.mean(np.log(v))))),
                   err=("abs_log2_error", "mean"))
              .reset_index())
