"""Turn the raw sweep CSVs into the tables and figures used in the report."""
import os, sys, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pandas as pd
import frontier, plots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

paths = sorted(glob.glob(os.path.join(RES, "sweep_*_k10.csv")))
if not paths:
    sys.exit("no sweep results yet")
df = frontier.load(paths)
df.to_csv(os.path.join(RES, "aggregated.csv"), index=False)

gap = frontier.default_gap(df)
gap.to_csv(os.path.join(RES, "default_gap.csv"), index=False)

pd.set_option("display.width", 200, "display.max_columns", 50)
print("=== configurations measured ===")
print(df.groupby(["dataset", "family"]).size().to_string())
print("\n=== timing variance across repeats (relative std of median latency) ===")
rel = (df["latency_ms_std"] / df["latency_ms"]).describe()
print(rel.to_string())
print("\n=== default vs frontier ===")
cols = ["dataset", "family", "default_params", "default_recall", "default_latency_ms",
        "best_recall_at_default_latency", "cheapest_latency_for_default_recall_ms",
        "latency_ms_at_recall_0.95", "params_at_recall_0.95"]
print(gap[[c for c in cols if c in gap.columns]].to_string(index=False))

plots.frontier_panels(df, os.path.join(FIG, "fig1_recall_latency_frontiers.png"),
                      "Recall-latency frontiers by index family (200k base vectors, k=10, single thread)")
plots.default_gap_bars(gap, os.path.join(FIG, "fig2_default_gap.png"))

chars_path = os.path.join(RES, "dataset_characteristics.json")
if os.path.exists(chars_path):
    chars = json.load(open(chars_path))
    plots.lid_vs_cost(gap, chars, os.path.join(FIG, "fig3_lid_vs_tuning_cost.png"))
    print("\n=== dataset characteristics ===")
    print(pd.DataFrame(chars)[["dataset", "ambient_dim", "intrinsic_dim_mle",
                               "lid_at_queries_mean", "inertia_log_slope"]].to_string(index=False))
print("\nfigures written to", FIG)
