"""Compute the candidate predictive features for every dataset.

These are the inputs the parameter-selection heuristic will use. Each is
computable from a sample of the base set without building any index, which is
the whole point: if they carry signal, a practitioner can compute them in
seconds instead of running a multi-hour parameter sweep.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import datasets, characterize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for name in ["sift-128-euclidean", "glove-100-angular", "nytimes-256-angular"]:
    base, q, metric, _ = datasets.load(name, n_base=200000, n_query=300)
    t0 = time.perf_counter()
    r = characterize.characterize(base, q, metric, n_sample=20000)
    r["dataset"] = name
    r["seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}),
          flush=True)
    rows.append(r)
json.dump(rows, open(os.path.join(ROOT, "results", "dataset_characteristics.json"), "w"), indent=2)
