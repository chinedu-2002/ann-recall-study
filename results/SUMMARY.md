# Headline numbers (Progress Report 1)

Base set 200,000 vectors per dataset, 300 queries, k = 10, single-threaded search,
3 timed repeats per configuration. Recall is tie-aware (distance-based).

## Harness validation

| dataset | identifier agreement@10 with published ground truth | tie-aware agreement@10 |
|---|---|---|
| sift-128-euclidean | 0.9991 | 1.0000 |
| glove-100-angular | 1.0000 | 1.0000 |
| nytimes-256-angular | 0.9856 | 1.0000 |

Every identifier disagreement (5 of 1000 on SIFT, 102 of 1000 on NYTimes) was an
exact distance tie, maximum distance gap 0.0.

## Library defaults vs the measured frontier

| dataset | family | default recall@10 | best recall at the same latency | latency to reach recall 0.95 |
|---|---|---|---|---|
| sift-128-euclidean | HNSW | 0.731 | 0.731 | 0.144 ms (3.3x default latency) |
| sift-128-euclidean | IVF-Flat | 0.528 | 0.773 | 0.334 ms |
| sift-128-euclidean | IVF-PQ | 0.242 | 0.384 | not reachable in grid |
| glove-100-angular | HNSW | 0.515 | 0.515 | 0.720 ms (17.6x) |
| glove-100-angular | IVF-Flat | 0.501 | 0.681 | 1.077 ms |
| glove-100-angular | IVF-PQ | 0.128 | 0.128 | not reachable in grid |
| nytimes-256-angular | HNSW | 0.679 | 0.679 | 1.792 ms (23.2x) |
| nytimes-256-angular | IVF-Flat | 0.482 | 0.774 | 15.100 ms |
| nytimes-256-angular | IVF-PQ | 0.291 | 0.298 | not reachable in grid |

No default configuration on any dataset reached recall 0.75. Mean default
recall@10 across the nine combinations is 0.455.

## Dataset characteristics (candidate heuristic features)

| dataset | ambient dim | intrinsic dim (MLE) | LID at queries | k-means inertia log-slope |
|---|---|---|---|---|
| sift-128-euclidean | 128 | 19.9 | 23.9 | -0.115 |
| glove-100-angular | 100 | 36.4 | 43.1 | -0.051 |
| nytimes-256-angular | 256 | 46.0 | 53.5 | -0.028 |

## Measurement stability

Relative standard deviation of median query latency across 3 repeats, over all
219 aggregated configurations: median 1.3%, mean 1.9%, maximum 10.5%.
