# Measuring the Recall Cost of Approximate Nearest Neighbor Search

Senior Seminar research project, Fisk University. Emmanuel C. Enenta.

## What this is

Production vector search almost never uses exact nearest neighbor search. It uses
an approximate index, accepts some recall loss in exchange for speed, and usually
leaves the index parameters at library defaults. This project measures what that
recall loss actually is, and tests whether good parameters can be predicted from
measurable properties of a dataset instead of found by exhaustive search.

Two questions:

1. Across index families and datasets, how much recall is sacrificed at a given
   latency budget, and how far do library defaults sit from the achievable frontier?
2. Can index parameters be selected predictively from dataset properties
   (dimensionality, intrinsic dimensionality, local intrinsic dimensionality,
   cluster structure) rather than by sweeping?

## Layout

```
src/datasets.py       loading, subsampling, normalization for angular metrics
src/groundtruth.py    exact brute-force kNN + identifier and tie-aware recall
src/indexes.py        uniform wrappers for HNSW, IVF-Flat, IVF-PQ
src/characterize.py   intrinsic dimension (Levina-Bickel MLE), LID, cluster structure
src/sweep.py          parameter sweep driver, streams results to CSV
src/frontier.py       Pareto frontier extraction, default-vs-frontier analysis
src/plots.py          figures
scripts/              validation, sweeps, analysis entry points
results/              measurement CSVs and validation JSON (committed)
figures/              generated figures
```

## Datasets

Standard ANN-Benchmarks HDF5 files, chosen to differ in dimensionality, distance
metric, and cluster structure:

| dataset | vectors | dim | metric | domain |
|---|---|---|---|---|
| sift-128-euclidean | 1,000,000 | 128 | euclidean | image descriptors |
| glove-100-angular | 1,183,514 | 100 | angular | word embeddings |
| nytimes-256-angular | 290,000 | 256 | angular | text documents |

## Reproducing

```bash
pip install -r requirements.txt
bash scripts/download_data.sh
python3 scripts/validate_groundtruth.py           # gate 1: exact search is correct
python3 scripts/validate_groundtruth_tieaware.py  # gate 1 under tie-aware recall
python3 scripts/investigate_ties.py               # explains the identifier disagreement
bash  scripts/run_all_sweeps.sh                   # the measurement study
python3 scripts/analyze.py                        # tables + figures
```

## Measurement notes

All search is single-threaded and timed per query, so latency is comparable
across index families on the same machine. Recall is machine-independent;
throughput is not, so published throughput numbers from other hardware are not
directly comparable and only recall is used as a correctness gate.
