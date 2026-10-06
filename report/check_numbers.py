"""Re-derive every number quoted in the report prose from the result files and
confirm the quoted string appears in the text. Exits non-zero on any mismatch."""
import os, sys, json, glob
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = lambda f: os.path.join(R, "results", f)
TEXT_FILE = sys.argv[1] if len(sys.argv) > 1 else "pr2_text_numbers.json"
PRESENT_ONLY = "--present-only" in sys.argv
T = json.load(open(os.path.join(R, "report", TEXT_FILE)))
text = " ".join(v if isinstance(v, str) else " ".join(" ".join(x) for x in v) for v in T.values())

chars = pd.DataFrame(json.load(open(res("dataset_characteristics.json")))).set_index("dataset")
gap = pd.read_csv(res("pr2_default_gap_6datasets.csv"))
S = pd.read_csv(res("pr2_heldout_summary.csv"))
ev = pd.read_csv(res("pr2_heldout_eval.csv"))
kp = pd.read_csv(res("pr2_kpass_default.csv"))
full = pd.read_csv(res("pr2_default_gap_fullsize.csv"))
nums = json.load(open(res("pr2_report_numbers.json")))
bv = pd.DataFrame(json.load(open(res("build_variance.json"))))
bo = pd.DataFrame(json.load(open(res("build_order.json"))))
dense = pd.read_csv(res("pr2_dense_aggregated.csv"))

checks = []
def c(label, quoted, value, fmt="{:.3f}"):
    checks.append((label, quoted, fmt.format(value) if not isinstance(value, str) else value))

def s(fam, m, col, t=0.95, feat="lid", tr="real+synthetic"):
    return float(S[(S.family == fam) & (S.method == m) & (S.target == t) & (S.feature == feat) & (S.training == tr)][col].iloc[0])

g = lambda ds, fam, col: float(gap[(gap.dataset == ds) & (gap.family == fam)][col].iloc[0])
c("mean default", "0.536", gap.default_recall.mean())
c("min default", "0.128", gap.default_recall.min())
c("max default", "0.930", gap.default_recall.max())
c("mnist hnsw", "0.929", g("mnist-784-euclidean", "HNSW", "default_recall"))
c("glove25 hnsw", "0.763", g("glove-25-angular", "HNSW", "default_recall"))
c("glove100 hnsw", "0.515", g("glove-100-angular", "HNSW", "default_recall"))
c("nytimes hnsw", "0.679", g("nytimes-256-angular", "HNSW", "default_recall"))
ivf = gap[gap.family == "IVF-Flat"]; gain = ivf.best_recall_at_default_latency - ivf.default_recall
c("ivf gain min", "0.171", gain.min()); c("ivf gain max", "0.291", gain.max())
c("r hnsw", "0.963", nums["fit_HNSW"]["r"]); c("r ivf", "0.959", nums["fit_IVF-Flat"]["r"])
c("a hnsw", "2.23", nums["fit_HNSW"]["a"], "{:.2f}"); c("b hnsw", "0.137", nums["fit_HNSW"]["b"])
c("a ivf", "1.65", nums["fit_IVF-Flat"]["a"], "{:.2f}"); c("b ivf", "0.124", nums["fit_IVF-Flat"]["b"])
c("doubling hnsw", "7.3", 1 / nums["fit_HNSW"]["b"], "{:.1f}"); c("doubling ivf", "8.1", 1 / nums["fit_IVF-Flat"]["b"], "{:.1f}")
c("syn max lid", "26", chars.loc["synthetic-id48-128"].lid_at_queries_mean, "{:.0f}")
c("const max cost", "4.79", s("HNSW", "constant-max", "geo_latency_vs_oracle"), "{:.2f}")
c("heur cost hnsw", "1.17", s("HNSW", "heuristic", "geo_latency_vs_oracle"), "{:.2f}")
c("heur cost ivf", "1.00", s("IVF-Flat", "heuristic", "geo_latency_vs_oracle"), "{:.2f}")
c("heur min hnsw", "0.924", s("HNSW", "heuristic", "min_recall"))
c("heur min ivf", "0.904", s("IVF-Flat", "heuristic", "min_recall"))
c("margin cost hnsw", "1.44", s("HNSW", "heuristic+margin", "geo_latency_vs_oracle"), "{:.2f}")
c("margin cost ivf", "1.42", s("IVF-Flat", "heuristic+margin", "geo_latency_vs_oracle"), "{:.2f}")
e = ev[(ev.feature == "lid") & (ev.training == "real+synthetic") & (ev.target == 0.95) & (ev.method == "heuristic+margin")]
c("glove100 margin miss", "0.937", float(e[(e.family == "HNSW") & (e.dataset == "glove-100-angular")].recall.iloc[0]))
c("glove25 margin miss", "0.938", float(e[(e.family == "IVF-Flat") & (e.dataset == "glove-25-angular")].recall.iloc[0]))
c("margin 0.90 ivf cost", "1.51", s("IVF-Flat", "heuristic+margin", "geo_latency_vs_oracle", 0.90), "{:.2f}")
c("constmax 0.90 ivf cost", "3.52", s("IVF-Flat", "constant-max", "geo_latency_vs_oracle", 0.90), "{:.2f}")
for fam, feat, q in (("HNSW", "lid", "0.72"), ("IVF-Flat", "lid", "0.47"), ("HNSW", "ambient_dim", "1.68"),
                     ("IVF-Flat", "ambient_dim", "1.31"), ("HNSW", "none", "1.97"), ("IVF-Flat", "none", "1.54")):
    c(f"ablation {fam} {feat}", q, s(fam, "heuristic", "mean_abs_log2_error", feat=feat), "{:.2f}")
c("real-only hnsw err", "0.83", s("HNSW", "heuristic", "mean_abs_log2_error", tr="real only"), "{:.2f}")
c("real-only ivf err", "0.50", s("IVF-Flat", "heuristic", "mean_abs_log2_error", tr="real only"), "{:.2f}")
k = lambda ds, kk, col="recall_at_ef10": float(kp[(kp.dataset == ds) & (kp.k == kk)][col].iloc[0])
c("k1 sift", "0.790", k("sift-128-euclidean", 1)); c("k10 sift", "0.729", k("sift-128-euclidean", 10))
c("k1 glove", "0.583", k("glove-100-angular", 1)); c("k10 glove", "0.503", k("glove-100-angular", 10))
c("k1 nyt", "0.670", k("nytimes-256-angular", 1)); c("k10 nyt", "0.604", k("nytimes-256-angular", 10))
c("k100 sift clamp", "0.94967", k("sift-128-euclidean", 100), "{:.5f}")
c("k100 glove clamp", "0.73077", k("glove-100-angular", 100), "{:.5f}")
c("k100 nyt clamp", "0.73317", k("nytimes-256-angular", 100), "{:.5f}")
c("k100 sift lat", "0.258", k("sift-128-euclidean", 100, "latency_ms")); c("k10 sift lat", "0.041", k("sift-128-euclidean", 10, "latency_ms"))
f = lambda ds, fam, col: float(full[(full.dataset == ds) & (full.family == fam)][col].iloc[0])
c("full sift hnsw", "0.696", f("sift-128-euclidean", "HNSW", "default_recall"))
c("full glove hnsw", "0.478", f("glove-100-angular", "HNSW", "default_recall"))
c("full sift ivf", "0.560", f("sift-128-euclidean", "IVF-Flat", "default_recall"))
c("full glove ivf", "0.535", f("glove-100-angular", "IVF-Flat", "default_recall"))
c("full sift best", "0.827", f("sift-128-euclidean", "IVF-Flat", "best_recall_at_default_latency"))
c("full glove best", "0.810", f("glove-100-angular", "IVF-Flat", "best_recall_at_default_latency"))
ny = bo[bo.dataset == "nytimes-256-angular"].recall_tieaware_ef10
for v in ny: c("nyt shuffled", f"{v:.3f}", v)
n2 = bv[(bv.dataset == "nytimes-256-angular") & (bv.build_threads == 2)].recall_tieaware_ef10
c("nyt 2t min", "0.590", n2.min()); c("nyt 2t max", "0.634", n2.max())
sift_all = list(bv[bv.dataset == "sift-128-euclidean"].recall_tieaware_ef10) + list(bo[bo.dataset == "sift-128-euclidean"].recall_tieaware_ef10)
c("sift rebuild min", "0.709", min(sift_all)); c("sift rebuild max", "0.743", max(sift_all))
c("dense nyt default", "0.604", float(dense[(dense.dataset == "nytimes-256-angular") & (dense.family == "HNSW") & (dense.query_param == 10)].recall.iloc[0]))

ci_old = pd.DataFrame(json.load(open(res("pr1_default_ci.json"))))
c("nyt query ci", "±0.044", "±{:.3f}".format(float(ci_old[(ci_old.dataset == "nytimes-256-angular") & (ci_old.family == "HNSW")].ci95.iloc[0])), "{}")
nyt_all = list(bv[bv.dataset == "nytimes-256-angular"].recall_tieaware_ef10) + list(ny) + [0.60433]
c("nyt spread", "0.089", max(nyt_all) - min(nyt_all))
for d, q in ((4, "4.3"), (8, "7.6"), (48, "22.8")):
    c(f"synthetic id{d}", q, chars.loc[f"synthetic-id{d}-128"].intrinsic_dim_mle, "{:.1f}")
bad = 0
for label, quoted, derived in checks:
    ok = quoted == derived and (quoted in text or PRESENT_ONLY)
    if not ok:
        bad += 1
        print(f"MISMATCH {label}: quoted {quoted}  derived {derived}  in_text {quoted in text}")
print(f"{len(checks) - bad} of {len(checks)} quoted numbers verified")
sys.exit(1 if bad else 0)
