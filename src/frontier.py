"""Pareto frontier extraction and default-configuration analysis.

The core question this module answers: given everything we measured, what is the
best latency achievable at each recall level, and how far from that frontier does
the library's own default configuration sit?
"""
import json
import numpy as np
import pandas as pd


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    g = (df.groupby(["dataset", "n_base", "dim", "metric", "family", "build_params",
                     "query_param", "k", "is_default"], as_index=False)
           .agg(recall=("recall_tieaware", "mean"),
                recall_id=("recall_id", "mean"),
                recall_std=("recall_tieaware", "std"),
                latency_ms=("latency_median_ms", "mean"),
                latency_ms_std=("latency_median_ms", "std"),
                latency_p95_ms=("latency_p95_ms", "mean"),
                build_seconds=("build_seconds", "mean"),
                memory_mb=("memory_mb", "mean")))
    return g


def pareto(df):
    """Keep configurations not dominated on both recall (higher) and latency (lower)."""
    d = df.sort_values("latency_ms").reset_index(drop=True)
    keep, best = [], -np.inf
    for i, r in d.iterrows():
        if r["recall"] > best:
            keep.append(i)
            best = r["recall"]
    return d.loc[keep].sort_values("recall").reset_index(drop=True)


def latency_at_recall(front, target):
    """Cheapest configuration on the frontier that reaches the target recall."""
    ok = front[front["recall"] >= target]
    if len(ok) == 0:
        return None
    return ok.loc[ok["latency_ms"].idxmin()]


def default_gap(df, targets=(0.90, 0.95, 0.99)):
    """For every (dataset, family): where the default sits, and what it costs."""
    out = []
    for (ds, fam), sub in df.groupby(["dataset", "family"]):
        dflt = sub[sub["is_default"] == 1]
        front = pareto(sub)
        rec = {"dataset": ds, "family": fam,
               "n_configs_measured": int(len(sub)),
               "n_on_frontier": int(len(front)),
               "best_recall_measured": float(sub["recall"].max())}
        if len(dflt):
            d0 = dflt.iloc[0]
            rec["default_params"] = f'{d0["build_params"]} / qp={d0["query_param"]}'
            rec["default_recall"] = float(d0["recall"])
            rec["default_latency_ms"] = float(d0["latency_ms"])
            # what the frontier achieves at the default's latency budget
            same_cost = front[front["latency_ms"] <= d0["latency_ms"] * 1.05]
            rec["best_recall_at_default_latency"] = (
                float(same_cost["recall"].max()) if len(same_cost) else None)
            # what it costs to match the default's recall
            match = latency_at_recall(front, d0["recall"])
            rec["cheapest_latency_for_default_recall_ms"] = (
                float(match["latency_ms"]) if match is not None else None)
        for t in targets:
            m = latency_at_recall(front, t)
            rec[f"latency_ms_at_recall_{t}"] = float(m["latency_ms"]) if m is not None else None
            rec[f"params_at_recall_{t}"] = (
                f'{m["build_params"]} / qp={m["query_param"]}' if m is not None else None)
        out.append(rec)
    return pd.DataFrame(out)
