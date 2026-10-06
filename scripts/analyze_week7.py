"""Week 7 analysis: does collection size (and hubness) improve the model?"""
import os, sys, glob, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
import datasets, heuristic2 as H2

RES = os.path.join(ROOT, "results")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
chars = pd.DataFrame(json.load(open(os.path.join(RES, "dataset_characteristics.json"))))
frames = []
for p in sorted(glob.glob(os.path.join(RES, "sweep_*_dense.csv"))):
    try:
        f = pd.read_csv(p)
        if len(f): frames.append(f)
    except pd.errors.EmptyDataError:
        pass
agg = H2.aggregate(pd.concat(frames, ignore_index=True))
agg.to_csv(os.path.join(RES, "week7_dense_aggregated.csv"), index=False)

orc = pd.concat([H2.oracle_rows(agg, chars, t) for t in (0.90, 0.95)], ignore_index=True)
orc.to_csv(os.path.join(RES, "week7_oracles.csv"), index=False)
print("=== oracle settings by collection size (target 0.95) ===")
o95 = orc[(orc.target == 0.95) & orc.dataset.isin(["sift-128-euclidean", "glove-100-angular"])]
print(o95.pivot_table(index=["dataset", "n_base"], columns="family", values="oracle_qp").to_string())

print("\n=== default (ef=10) recall by collection size ===")
d = agg[(agg.family == "HNSW") & (agg.query_param == 10) & agg.dataset.isin(["sift-128-euclidean", "glove-100-angular"])]
print(d.pivot_table(index="n_base", columns="dataset", values="recall").round(4).to_string())

print("\n=== hubness ===")
print(chars[["dataset", "lid_at_queries_mean", "hubness_skew_k10"]].round(2).to_string(index=False))

REAL = datasets.REAL_DATASETS
evs = []
for fam in ("HNSW", "IVF-Flat"):
    for t in (0.90, 0.95):
        for m in H2.MODELS:
            e = H2.evaluate(agg, orc[orc.target == t], fam, m, REAL, "primary"); e["test"] = "primary"; evs.append(e)
            e = H2.evaluate(agg, orc[orc.target == t], fam, m, ["sift-128-euclidean", "glove-100-angular"], "all")
            e["test"] = "all sizes"; evs.append(e)
ev = pd.concat(evs, ignore_index=True)
ev.to_csv(os.path.join(RES, "week7_heldout_eval.csv"), index=False)
summ = ev.groupby("test").apply(H2.summarize, include_groups=False).reset_index(level=0).reset_index(drop=True)
summ.to_csv(os.path.join(RES, "week7_heldout_summary.csv"), index=False)
print("\n=== held-out, six real datasets at their week 5-6 size ===")
print(summ[summ.test == "primary"].drop(columns="test").round(3).to_string(index=False))
print("\n=== held-out, SIFT and GloVe-100 at all five sizes ===")
print(summ[summ.test == "all sizes"].drop(columns="test").round(3).to_string(index=False))

# coefficients of the full-data fits, for the report
out = {}
for fam in ("HNSW", "IVF-Flat"):
    o = orc[(orc.family == fam) & (orc.target == 0.95)]
    for m, feats in H2.MODELS.items():
        X = o[feats].values if feats else np.empty((len(o), 0))
        coef, sigma = H2.fit(X, o["oracle_qp"].values)
        out[f"{fam}|{m}"] = {"coef": [float(c) for c in coef], "features": feats, "rms_log2": sigma, "n_rows": int(len(o))}
json.dump(out, open(os.path.join(RES, "week7_fits.json"), "w"), indent=2)
for k, v in out.items():
    print(k, [round(c, 3) for c in v["coef"]], "rms", round(v["rms_log2"], 3), "rows", v["n_rows"])
