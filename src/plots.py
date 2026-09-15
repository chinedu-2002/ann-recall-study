"""Figures for the ANN recall study.

Design choices follow the project's visualization conventions: one hue per index
family assigned in fixed order and never cycled, a single axis per panel, thin
marks, recessive grid, log-scaled latency because the measured range spans three
orders of magnitude, and direct labels on the library-default configurations so
identity is never carried by color alone.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import frontier

FAMILY_COLOR = {"HNSW": "#2a78d6", "IVF-Flat": "#eb6834", "IVF-PQ": "#1baf7a"}
FAMILY_MARKER = {"HNSW": "o", "IVF-Flat": "s", "IVF-PQ": "^"}
# label offsets chosen per family so the three default annotations never collide
LABEL_OFFSET = {"HNSW": (12, -24), "IVF-Flat": (14, 12), "IVF-PQ": (12, -26)}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"

DATASET_LABEL = {
    "sift-128-euclidean": "SIFT-128 (image descriptors, euclidean)",
    "glove-100-angular": "GloVe-100 (word embeddings, angular)",
    "nytimes-256-angular": "NYTimes-256 (text documents, angular)",
}


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="major", color=MUTED, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def frontier_panels(df, out_path, title, stacked=False, page_width=6.6):
    """Recall-latency frontiers, one panel per dataset.

    `stacked` lays the panels out vertically at true print width, so that when
    the figure is embedded in the report at `page_width` inches nothing is scaled
    down and the tick labels stay at their intended point size.
    """
    datasets = [d for d in DATASET_LABEL if d in set(df["dataset"])]
    n = len(datasets)
    if stacked:
        fig, axes = plt.subplots(n, 1, figsize=(page_width, 2.85 * n), squeeze=False)
        flat = [axes[i][0] for i in range(n)]
    else:
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), squeeze=False)
        flat = list(axes[0])
    fig.patch.set_facecolor("white")
    for ax, ds in zip(flat, datasets):
        sub = df[df["dataset"] == ds]
        _style(ax)
        for fam in ["HNSW", "IVF-Flat", "IVF-PQ"]:
            f = sub[sub["family"] == fam]
            if not len(f):
                continue
            c = FAMILY_COLOR[fam]
            ax.scatter(f["latency_ms"], f["recall"], s=9, color=c, alpha=0.22,
                       linewidths=0, zorder=2)
            fr = frontier.pareto(f)
            ax.plot(fr["latency_ms"], fr["recall"], color=c, linewidth=2,
                    marker=FAMILY_MARKER[fam], markersize=4.5, label=fam, zorder=3)
            d0 = f[f["is_default"] == 1]
            if len(d0):
                r = d0.iloc[0]
                ax.scatter([r["latency_ms"]], [r["recall"]], s=115, facecolors="none",
                           edgecolors=c, linewidths=2.0, zorder=5)
                ax.annotate(f"{fam} default\nrecall {r['recall']:.3f}",
                            (r["latency_ms"], r["recall"]),
                            textcoords="offset points", xytext=LABEL_OFFSET[fam],
                            fontsize=7.2, color=INK2, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                      ec="none", alpha=0.82))
        ax.set_xscale("log")
        ax.set_xlabel("median query latency (ms, log scale)", fontsize=8.5, color=INK2)
        ax.set_ylabel("recall@10 (tie-aware)", fontsize=8.5, color=INK2)
        ax.set_title(DATASET_LABEL[ds], fontsize=9.5, color=INK, pad=8)
        ax.set_ylim(-0.03, 1.04)
        ax.axhline(0.95, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
        ax.text(ax.get_xlim()[0], 0.955, " recall 0.95", fontsize=7, color=INK2, va="bottom")
    flat[0].legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    fig.suptitle(title, fontsize=10.5, color=INK, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def default_gap_bars(gap_df, out_path):
    """How much recall the library default leaves on the table at its own latency."""
    d = gap_df.dropna(subset=["default_recall", "best_recall_at_default_latency"]).copy()
    d["label"] = d["dataset"].str.split("-").str[0] + "\n" + d["family"]
    d = d.sort_values(["family", "dataset"])
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    fig.patch.set_facecolor("white")
    _style(ax)
    w = 0.38
    ax.bar(x - w / 2, d["default_recall"], w, color="#b8b7b0", label="library default",
           zorder=3)
    ax.bar(x + w / 2, d["best_recall_at_default_latency"], w,
           color=[FAMILY_COLOR[f] for f in d["family"]],
           label="best tuned config at the same latency", zorder=3)
    for xi, (a, b) in enumerate(zip(d["default_recall"], d["best_recall_at_default_latency"])):
        ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7, color=INK2)
        ax.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(d["label"], fontsize=7.5, color=INK2)
    ax.set_ylabel("recall@10 (tie-aware)", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_title("Recall left on the table by default parameters, at equal latency",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              labelcolor=INK2, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def lid_vs_cost(gap_df, chars, out_path, target=0.95):
    """Preliminary signal check for the heuristic: does query-point LID track the
    latency needed to reach a fixed recall target?"""
    col = f"latency_ms_at_recall_{target}"
    d = gap_df[gap_df["family"] == "HNSW"].dropna(subset=[col]).copy()
    lid = {c["dataset"]: c["lid_at_queries_mean"] for c in chars}
    idim = {c["dataset"]: c["intrinsic_dim_mle"] for c in chars}
    d["lid"] = d["dataset"].map(lid)
    d["idim"] = d["dataset"].map(idim)
    d = d.sort_values("lid")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig.patch.set_facecolor("white")
    _style(ax)
    ax.plot(d["lid"], d[col], color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    ax.scatter(d["lid"], d[col], s=70, color=FAMILY_COLOR["HNSW"], zorder=3)
    for _, r in d.iterrows():
        ax.annotate(f'{r["dataset"].split("-")[0]}\nintrinsic dim {r["idim"]:.0f}',
                    (r["lid"], r[col]), textcoords="offset points", xytext=(10, -4),
                    fontsize=7.5, color=INK2)
    ax.set_xlabel("mean local intrinsic dimensionality at the query points",
                  fontsize=8.5, color=INK2)
    ax.set_ylabel(f"HNSW latency needed to reach recall@10 = {target} (ms)",
                  fontsize=8.5, color=INK2)
    ax.set_title("Does a cheap dataset statistic predict tuning cost?",
                 fontsize=10.5, color=INK, pad=10)
    ax.set_xlim(d["lid"].min() - 6, d["lid"].max() + 16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path
