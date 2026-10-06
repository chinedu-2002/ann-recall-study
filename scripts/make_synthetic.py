"""Generate synthetic datasets with a controlled intrinsic dimension.

Each dataset draws latent points from a mixture of 20 Gaussian clusters in d
dimensions, pushes them through a fixed random nonlinear map (a two-layer tanh
network) into 128 ambient dimensions, and adds a small amount of isotropic noise.
The cluster layout, the network width and the noise level are identical across
datasets; only the latent dimension d changes. That isolates intrinsic
dimensionality as the variable of interest.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import datasets

D_AMBIENT, N_BASE, N_QUERY, N_CLUSTERS, HIDDEN, NOISE = 128, 200_000, 1_000, 20, 256, 0.02


def make(d, seed=0):
    rng = np.random.default_rng(1000 + d)
    centers = rng.normal(scale=2.0, size=(N_CLUSTERS, d))
    W1 = rng.normal(scale=1.0 / np.sqrt(d), size=(d, HIDDEN))
    b1 = rng.normal(scale=0.1, size=HIDDEN)
    W2 = rng.normal(scale=1.0 / np.sqrt(HIDDEN), size=(HIDDEN, D_AMBIENT))

    def sample(n):
        z = centers[rng.integers(0, N_CLUSTERS, n)] + rng.normal(size=(n, d))
        x = np.tanh(z @ W1 + b1) @ W2
        return (x + rng.normal(scale=NOISE, size=x.shape)).astype(np.float32)

    return sample(N_BASE), sample(N_QUERY)


if __name__ == "__main__":
    for d in datasets.SYNTHETIC_IDS:
        base, q = make(d)
        out = os.path.join(datasets.DATA_DIR, f"synthetic-id{d}-128.npz")
        np.savez(out, base=base, queries=q)
        print(f"id={d:3d}  base {base.shape}  queries {q.shape}  -> {os.path.basename(out)}", flush=True)
