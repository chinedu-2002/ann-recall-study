"""Progress Report 2 analysis: heuristic fit and held-out evaluation, the default
gap on six real datasets, k passes, and the full-size check. Every table and
figure in the report comes from this script and the committed CSVs."""
import os, sys, glob, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
import datasets, frontier, heuristic as H, plots

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
REAL, SYN = datasets.REAL_DATASETS, datasets.SYNTHETIC_DATASETS
chars = pd.DataFrame(json.load(open(os.path.join(RES, "dataset_characteristics.json"))))
Z = 1.96
report = {}


def read_all(paths):
    """Concatenate CSVs, skipping any file still being created by a running sweep."""
    frames = []
    for p in paths:
        try:
            f = pd.read_csv(p)
            if len(f):
                frames.append(f)
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else None


def ci(std, n):
    return Z * std / np.sqrt(n)


# ---------------------------------------------------------------- 1. heuristic
dense_paths = sorted(glob.glob(os.path.join(RES, "sweep_*_dense.csv")))
if dense_paths:
    dense = H.aggregate(read_all(dense_paths))
    dense.to_csv(os.path.join(RES, "pr2_dense_aggregated.csv"), index=False)
    have = set(dense["dataset"])
    print(f"dense sweeps available for {len(have)} datasets")

    evs, orcs = [], []
    for target in (0.90, 0.95):
        for fam, dflt in (("HNSW", 10), ("IVF-Flat", 1)):
            sub = dense[dense["family"] == fam]
            o = H.oracle_table(sub, target); o["family"] = fam; orcs.append(o)
            real_here = [d for d in REAL if d in have]
            pool_all = [d for d in REAL + SYN if d in have]
            for feat in ("lid", "intrinsic_dim", "ambient_dim", None):
                e = H.evaluate(sub, chars, fam, target, feat, real_here, pool_all, dflt)
                e["training"] = "real+synthetic"; evs.append(e)
            e = H.evaluate(sub, chars, fam, target, "lid", real_here, real_here, dflt)
            e["training"] = "real only"; evs.append(e)
    ev = pd.concat(evs, ignore_index=True)
    ev["feature"] = ev["feature"].fillna("none")
    orc = pd.concat(orcs, ignore_index=True)
    ev.to_csv(os.path.join(RES, "pr2_heldout_eval.csv"), index=False)
    orc.to_csv(os.path.join(RES, "pr2_oracle_settings.csv"), index=False)
    summ = (ev.groupby(["training"]).apply(H.summarize, include_groups=False)
              .reset_index(level=0).reset_index(drop=True))
    summ.to_csv(os.path.join(RES, "pr2_heldout_summary.csv"), index=False)

    print("\n=== oracle settings (smallest setting reaching target) ===")
    piv = orc.pivot_table(index="dataset", columns=["family", "target"], values="oracle_qp")
    print(piv.reindex([d for d in REAL + SYN if d in piv.index]).to_string())

    print("\n=== held-out summary, feature = LID, training = real+synthetic ===")
    s1 = summ[(summ["training"] == "real+synthetic") & (summ["feature"] == "lid")]
    print(s1.drop(columns=["feature"]).to_string(index=False))

    print("\n=== feature ablation (heuristic only): mean |log2 error| and hit rate ===")
    s2 = summ[(summ["training"] == "real+synthetic") & (summ["method"] == "heuristic")]
    print(s2[["family", "target", "feature", "hit_rate", "geo_latency_vs_oracle",
              "mean_abs_log2_error"]].to_string(index=False))

    print("\n=== training-set ablation (LID heuristic) ===")
    s3 = summ[(summ["feature"] == "lid") & (summ["method"] == "heuristic")]
    print(s3[["training", "family", "target", "hit_rate", "geo_latency_vs_oracle",
              "mean_abs_log2_error"]].to_string(index=False))

    print("\n=== per-dataset held-out detail (LID, real+synthetic, target 0.95) ===")
    d = ev[(ev["feature"] == "lid") & (ev["training"] == "real+synthetic") & (ev["target"] == 0.95)]
    print(d[["family", "dataset", "method", "setting", "raw_prediction", "recall",
             "latency_ms", "latency_vs_oracle"]].to_string(index=False))

    # figures
    fits = {}
    for fam in ("HNSW", "IVF-Flat"):
        o = orc[(orc["family"] == fam) & (orc["target"] == 0.95)].dropna(subset=["oracle_qp"])
        lid = o["dataset"].map(chars.set_index("dataset")["lid_at_queries_mean"])
        fits[fam] = H.fit(lid.values, o["oracle_qp"].values)
        r = np.corrcoef(lid.values, np.log2(o["oracle_qp"].values))[0, 1]
        resid = np.log2(o["oracle_qp"].values) - (fits[fam][0] + fits[fam][1] * lid.values)
        sigma = float(np.sqrt(np.mean(resid ** 2)))
        print(f"\nfit on all datasets, {fam}, target 0.95: log2(setting) = "
              f"{fits[fam][0]:.3f} + {fits[fam][1]:.4f} * LID   (Pearson r = {r:.3f}, n = {len(o)})")
        report[f"fit_{fam}"] = {"a": fits[fam][0], "b": fits[fam][1], "r": float(r), "n": int(len(o)),
                                "sigma_log2": sigma, "target": 0.95}
    plots.heuristic_fit({f: orc[(orc["family"] == f) & (orc["target"] == 0.95)] for f in ("HNSW", "IVF-Flat")},
                        chars, fits, os.path.join(FIG, "fig4_heuristic_fit.png"), 0.95)
    lid_ev = ev[(ev["feature"] == "lid") & (ev["training"] == "real+synthetic")]
    plots.heldout_panels(lid_ev, os.path.join(FIG, "fig5_heldout_hnsw.png"), 0.95, "HNSW")
    plots.heldout_panels(lid_ev, os.path.join(FIG, "fig5b_heldout_ivf.png"), 0.95, "IVF-Flat")

# ------------------------------------------------------ 2. default gap, 6 sets
k10 = sorted(glob.glob(os.path.join(RES, "sweep_*_200000_k10.csv")))
if k10:
    df = frontier.load(k10)
    gap = frontier.default_gap(df)
    raw = read_all(k10)
    if "recall_q_std" in raw:
        qs = (raw[raw["is_default"] == 1].groupby(["dataset", "family"])
                 .agg(q_std=("recall_q_std", "mean"), n=("n_query", "first")).reset_index())
        gap = gap.merge(qs, on=["dataset", "family"], how="left")
        gap["default_recall_ci95"] = ci(gap["q_std"], gap["n"])
    gap.to_csv(os.path.join(RES, "pr2_default_gap_6datasets.csv"), index=False)
    print("\n=== default gap across datasets (PR1 grid, k = 10) ===")
    cols = ["dataset", "family", "default_recall", "default_recall_ci95", "best_recall_at_default_latency",
            "latency_ms_at_recall_0.95"]
    print(gap[[c for c in cols if c in gap]].to_string(index=False))
    print("mean default recall over", len(gap), "combinations:", round(gap["default_recall"].mean(), 4),
          " max:", round(gap["default_recall"].max(), 4))
    report["default_gap"] = {"n": int(len(gap)), "mean": float(gap["default_recall"].mean()),
                             "max": float(gap["default_recall"].max())}
    plots.frontier_panels(df, os.path.join(FIG, "fig1_six_datasets.png"),
                          "Recall-latency frontiers, six datasets (k=10, single-thread search)",
                          stacked="grid", page_width=6.35, panel_h=2.75)
    plots.default_gap_dumbbell(gap, os.path.join(FIG, "fig2_default_gap_six.png"), chars)

# ------------------------------------------------------------- 3. k passes
kp = sorted(glob.glob(os.path.join(RES, "sweep_*_kpass.csv")))
if kp:
    kraw = read_all(kp)
    kagg = H.aggregate(kraw)
    # merge k = 10 from the dense sweep so the figure shows all three k
    if dense_paths:
        k10d = dense[(dense["family"] == "HNSW") & (dense["dataset"].isin(kagg["dataset"].unique()))]
        kagg = pd.concat([kagg, k10d], ignore_index=True)
    h = kagg[(kagg["family"] == "HNSW")]
    print("\n=== HNSW default build at ef = 10 across k (95% CI over queries) ===")
    rows = []
    for (ds, k), g in h[h["query_param"] == 10].groupby(["dataset", "k"]):
        r = g.iloc[0]
        rows.append({"dataset": ds, "k": k, "recall_at_ef10": r["recall"],
                     "ci95": ci(r["recall_q_std"], r["n_query"]), "latency_ms": r["latency_ms"]})
    kt = pd.DataFrame(rows).sort_values(["dataset", "k"])
    print(kt.to_string(index=False))
    kt.to_csv(os.path.join(RES, "pr2_kpass_default.csv"), index=False)
    print("\n=== clamp check at k = 100: ef = 10 vs ef = 100 ===")
    for ds, g in h[h["k"] == 100].groupby("dataset"):
        a = g[g["query_param"] == 10]["recall"].values
        b = g[g["query_param"] == 100]["recall"].values
        print(f"{ds:24s} ef10 {a[0]:.5f}   ef100 {b[0]:.5f}   identical: {bool(np.isclose(a[0], b[0]))}")
    ivf = kagg[kagg["family"] == "IVF-Flat"]
    print("\n=== IVF-Flat default (nlist 100, nprobe 1) across k ===")
    print(ivf[(ivf["query_param"] == 1) & (ivf["build_params"] == '{"nlist": 100}')]
          [["dataset", "k", "recall", "latency_ms"]].sort_values(["dataset", "k"]).to_string(index=False))
    plots.kpass_panels(h, os.path.join(FIG, "fig6_kpass.png"),
                       [d for d in ("sift-128-euclidean", "glove-100-angular", "nytimes-256-angular")
                        if d in set(h["dataset"])])

# ------------------------------------------------------------- 4. full size
sc = sorted(glob.glob(os.path.join(RES, "sweep_*_full_scale.csv")))
if sc:
    sraw = read_all(sc)
    sdf = frontier.load(sc)
    sgap = frontier.default_gap(sdf)
    qs = (sraw[sraw["is_default"] == 1].groupby(["dataset", "family"])
             .agg(q_std=("recall_q_std", "mean"), n=("n_query", "first"),
                  n_base=("n_base", "first")).reset_index())
    sgap = sgap.merge(qs, on=["dataset", "family"], how="left")
    sgap["default_recall_ci95"] = ci(sgap["q_std"], sgap["n"])
    sgap.to_csv(os.path.join(RES, "pr2_default_gap_fullsize.csv"), index=False)
    print("\n=== full-size default gap ===")
    print(sgap[["dataset", "family", "n_base", "default_recall", "default_recall_ci95",
                "best_recall_at_default_latency", "latency_ms_at_recall_0.95"]].to_string(index=False))

json.dump(report, open(os.path.join(RES, "pr2_report_numbers.json"), "w"), indent=2)
print("\nfigures written to", FIG)
