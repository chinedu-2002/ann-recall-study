"""Schematic of how IVF and HNSW answer a query (figures/fig0_methods.png).

Toy 2-D data only; it illustrates the mechanism, not a measurement.
Left: IVF splits space into cells around k-means centroids and scans only the
nprobe cells nearest the query, so a true neighbor just across a cell border is
missed at nprobe = 1 and found at nprobe = 3.
Right: HNSW stores the data as a stack of graphs. Search enters at the sparse top
layer, walks greedily toward the query, then drops down; on the bottom layer it keeps
a candidate list of size ef.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "fig0_methods.png")
BLUE, ORANGE, INK, INK2, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#b8b7b0"
rng = np.random.default_rng(7)

fig = plt.figure(figsize=(9.0, 3.3), dpi=220)
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.12)

# ---------------------------------------------------------------- IVF panel
ax = fig.add_subplot(gs[0])
cent = np.array([[0.18, 0.78], [0.50, 0.85], [0.83, 0.75], [0.15, 0.40], [0.47, 0.50],
                 [0.80, 0.42], [0.25, 0.10], [0.58, 0.15], [0.88, 0.10]])
pts = np.vstack([c + rng.normal(0, 0.075, (18, 2)) for c in cent]).clip(0.01, 0.99)
gx, gy = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400))
_, cell = cKDTree(cent).query(np.c_[gx.ravel(), gy.ravel()])
cell = cell.reshape(gx.shape)
q = np.array([0.52, 0.60])
order = np.argsort(np.linalg.norm(cent - q, axis=1))
probe1, probe3 = order[:1], order[:3]
shade = np.full(gx.shape, 0.0)
shade[np.isin(cell, probe3)] = 0.45
shade[np.isin(cell, probe1)] = 1.0
ax.imshow(shade, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=3.2, alpha=0.9)
ax.contour(gx, gy, cell, levels=np.arange(len(cent)) + 0.5, colors=MUTED, linewidths=0.6)
ax.scatter(cent[:, 0], cent[:, 1], s=34, marker="+", c=INK, lw=1.0)
# the true nearest neighbour, placed just across the border of the nearest cell
# walk from the query toward the 2nd-nearest centroid until we cross into its cell
c2 = cent[order[1]]
for t_ in np.linspace(0, 1, 400):
    x = q + t_ * (c2 - q)
    if cKDTree(cent).query(x)[1] == order[1]:
        break
nn = q + (t_ + 0.06) * (c2 - q)
# make it the true nearest neighbour: drop any data point closer to the query
pts = pts[np.linalg.norm(pts - q, axis=1) > np.linalg.norm(nn - q) * 1.15]
ax.scatter(pts[:, 0], pts[:, 1], s=5, c=INK2, lw=0)
ax.scatter(*q, s=110, marker="*", c=ORANGE, edgecolor=INK, lw=0.5, zorder=5)
ax.scatter(*nn, s=46, facecolor="none", edgecolor=ORANGE, lw=1.4, zorder=5)
BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
ax.annotate("query", q, (q[0] - 0.40, q[1] - 0.14), fontsize=7.5, color=INK, bbox=BOX,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.annotate("true nearest neighbor", nn, (nn[0] + 0.10, nn[1] + 0.06), fontsize=7.5, color=INK, bbox=BOX,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color(MUTED)
ax.set_title("IVF: scan only the nprobe nearest cells", fontsize=8.5, color=INK, loc="left")
ax.text(0.0, -0.10, "dark cell = nprobe 1 (misses the neighbor)\nlight cells = nprobe 3 (finds it)",
        transform=ax.transAxes, fontsize=7.2, color=INK2, va="top")

# ---------------------------------------------------------------- HNSW panels
base = rng.uniform(0.04, 0.96, (60, 2))
lvl = np.zeros(len(base), int)
lvl[rng.choice(len(base), 20, replace=False)] = 1
lvl[rng.choice(np.where(lvl == 1)[0], 7, replace=False)] = 2
q2 = np.array([0.80, 0.30])


def knn_edges(p, m):
    t = cKDTree(p)
    out = []
    for i, x in enumerate(p):
        for j in t.query(x, m + 1)[1][1:]:
            out.append((i, j))
    return out


top = np.where(lvl == 2)[0]
entry = int(top[np.argmax(np.linalg.norm(base[top] - q2, axis=1))])
cur = entry
for li, layer in enumerate([2, 1, 0]):
    a = fig.add_subplot(gs[li + 1])
    idx = np.where(lvl >= layer)[0]
    if entry not in idx:
        idx = np.append(idx, entry)
    p = base[idx]
    for i, j in knn_edges(p, 3 if len(p) > 4 else len(p) - 1):
        a.plot(*p[[i, j]].T, color=MUTED, lw=0.5, zorder=1)
    a.scatter(p[:, 0], p[:, 1], s=9 if layer == 0 else 16, c=INK2, lw=0, zorder=2)
    # greedy walk on this layer's graph, starting where the previous layer ended
    t = cKDTree(p)
    nbrs = {i: set(t.query(x, min(4, len(p)))[1][1:]) for i, x in enumerate(p)}
    pos = int(np.where(idx == cur)[0][0])
    path = [pos]
    while True:
        best = min(nbrs[pos] | {pos}, key=lambda k: np.linalg.norm(p[k] - q2))
        if best == pos:
            break
        pos = best
        path.append(pos)
    for u, v in zip(path[:-1], path[1:]):
        a.annotate("", p[v], p[u], arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3,
                                                     mutation_scale=8), zorder=4)
    a.scatter(*p[path[0]], s=30, c=BLUE, zorder=5)
    cur = int(idx[pos])
    a.scatter(*q2, s=110, marker="*", c=ORANGE, edgecolor=INK, lw=0.5, zorder=6)
    if layer == 0:
        d = np.linalg.norm(p - q2, axis=1)
        beam = np.argsort(d)[:6]
        a.scatter(p[beam, 0], p[beam, 1], s=70, facecolor="none", edgecolor=BLUE, lw=1.1, zorder=7)
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(MUTED)
    names = {2: "HNSW layer 2 (sparse)", 1: "layer 1", 0: "layer 0 (every vector)"}
    a.set_title(names[layer], fontsize=8.5, color=INK, loc="left")
fig.text(0.415, 0.06, "arrows = greedy walk toward the query, layer by layer.  "
         "Circled = the ef candidate list (ef = 6).", fontsize=7.2, color=INK2, va="bottom")
fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.17)
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
