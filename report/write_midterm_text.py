"""Prose for the Midterm Project Report (weeks 1 to 7). Every decimal quoted here
comes from the result files in results/."""
import json, os
T = {}

T["overview"] = [
 ("Title.", "Measuring the Recall Cost of Approximate Nearest Neighbor Search: A Parameter Selection Heuristic "
  "for Vector Databases."),
 ("Problem.", "Semantic search, recommendation and retrieval-augmented generation all rely on approximate "
  "nearest neighbor (ANN) indexes, which trade some accuracy for speed through tunable settings. Developers "
  "usually leave those settings at library defaults, and because an approximate index still returns "
  "plausible results when it misses the true neighbors, the lost recall produces no error and no warning."),
 ("Goals.", "First, measure how much recall the library defaults give up across index types and datasets. "
  "Second, map the best achievable recall at each latency so the defaults can be compared against it. Third, "
  "build and test a model that predicts the setting a dataset needs from cheap properties of the data, instead "
  "of an exhaustive sweep. Fourth, make all of it reproducible from a public repository."),
 ("Expected outcome.", "A final research report, a public code repository with the measurement harness and "
  "every raw result, the prediction model with a held-out evaluation, and a final presentation."),
]

T["work_intro"] = (
 "Over seven weeks the project grew to 2,003 lines of Python, 12 datasets and 4,101 timed measurements. "
 "The work fell into four phases.")

T["phases"] = [
 ("Weeks 1 and 2: topic and proposal",
  "I narrowed a broad interest in vector search to one measurable question, chose the ANN-Benchmarks datasets "
  "because they ship with published ground truth, and wrote the proposal with explicit success criteria: the "
  "harness must reproduce published results, the study must cover every index and dataset pair, and the model "
  "must be judged on data it has not seen."),
 ("Weeks 3 and 4: measurement harness and first study",
  "I built the harness in Python with FAISS (IVF-Flat and IVF-PQ indexes), hnswlib (HNSW graphs), NumPy, "
  "scikit-learn, pandas, h5py and matplotlib: modules for loading data, exact ground truth, uniform index "
  "wrappers, dataset characterization, parameter sweeps and Pareto-frontier analysis. Before trusting any "
  "number I checked my exact search against the published neighbors on the full collections. Agreement was "
  "0.9991 on SIFT and only 0.9856 on NYTimes; all 102 NYTimes disagreements turned out to be exact distance "
  "ties between duplicate vectors, so I defined recall on distances, which brought agreement to 1.0000 on "
  "every dataset. I then ran 657 timed measurements over 219 configurations on SIFT, GloVe-100 and NYTimes. "
  "The mean recall@10 of the nine library defaults was 0.455."),
 ("Weeks 5 and 6: more data and the prediction model",
  "I added Fashion-MNIST, MNIST and GloVe-25, plus six synthetic datasets with a known intrinsic dimension "
  "(4 to 48), which also let me check the dimensionality estimator: it reads 4.3 for a true 4 and 7.6 for a "
  "true 8, but only 22.8 for a true 48. Fine-grained sweeps found, for each dataset, the cheapest setting that "
  "reaches a target recall. I built the model (log2 of the needed setting as a linear function of local "
  "intrinsic dimensionality, LID) and tested it leave-one-dataset-out against the library default, a safe "
  "constant, a median constant and the oracle. I also ran the defaults at k = 1 and k = 100 and on the full "
  "million-vector collections, added 95% confidence intervals, measured how much recall changes when the same "
  "index is rebuilt, and wrote two live demonstration scripts. This phase added 2,556 measurements."),
 ("Week 7: collection size and hubness",
  "The weeks 5 and 6 results suggested collection size was tangled up with LID, so I swept SIFT and GloVe-100 at "
  "50,000, 100,000, 500,000 and full size (200,000 already existed), 888 more measurements. I also added "
  "hubness, the skew of how often each point appears in other points' neighbor lists, as a feature for all "
  "twelve datasets, extended the model to several features, and repeated the held-out test with every size of "
  "the held-out dataset removed from training."),
]

T["gap_text"] = (
 "With six datasets the default gap depends on the index type. IVF-Flat leaves recall on the table everywhere: "
 "at the default's own latency a tuned index gains between 0.171 and 0.291 recall on every dataset. The HNSW "
 "default falls as LID rises, from 0.93 on MNIST and Fashion-MNIST to 0.515 on GloVe-100 (Figure 1).")
T["fit_text"] = (
 "Across all twelve datasets, the log of the smallest setting that reaches recall 0.95 tracks query-point LID "
 "with a correlation of 0.963 for HNSW and 0.959 for IVF-Flat (Figure 2). Synthetic and real datasets fall on "
 "the same line.")
T["heldout_text"] = (
 "Table 1 shows the weeks 5 and 6 held-out test at a target of 0.95. The library default never reaches the "
 "target. The only safe constant costs 4.79 times the oracle's latency on HNSW, while the model with a safety "
 "margin reaches the target on five of six held-out datasets at 1.44 times.")
T["size_text"] = (
 "On HNSW, collection size matters on its own (Figure 3). From 50,000 vectors to the full collection, the "
 "default's recall falls from 0.805 to 0.689 on SIFT and from 0.562 to 0.464 on GloVe-100, and the ef needed "
 "for recall 0.95 doubles on SIFT (32 to 64) and quadruples on GloVe-100 (256 to 1,024). The fitted size term "
 "says the needed ef grows by a factor of 1.31 every time the collection doubles. Adding it to the model cuts "
 "the held-out HNSW error from 0.79 to 0.58 log2 units, and lowers the margin version's cost from 1.55 to 1.30 "
 "times the oracle at the same five-of-six hit rate (Table 2). It also fixes the MNIST over-prediction from "
 "weeks 5 and 6: the predicted ef drops from 24 to 16, exactly the true value. For IVF-Flat with a fixed number "
 "of lists the size term is near zero (-0.08) and does not help. Hubness confirms that GloVe-100 is unusual, "
 "with a skew of 5.25 against at most 2.84 for every other dataset, but adding it barely changes the error "
 "(0.58 to 0.55) and GloVe-100 is still under-predicted (256 against a true 512). The LID-only figures in Table 2 "
 "differ slightly from Table 1 because the training set now includes the eight new size rows.")

T["plan_compare"] = (
 "The proposal scheduled the harness for weeks 1 to 3, the parameter sweeps for weeks 4 to 8, the heuristic "
 "for weeks 9 to 11 and the write-up for weeks 12 to 14. Table 3 compares that plan with what happened. The "
 "project is ahead of schedule: the sweeps are finished and the model's first version arrived about four weeks "
 "early.")
T["changes"] = [
 ("Baseline check.", "The proposal planned to reproduce a published recall-latency curve. Latency depends on "
  "the hardware, so I used the published ground truth instead, which tests correctness exactly and on every "
  "dataset."),
 ("Six real datasets plus six synthetic, instead of three.", "A held-out test with three datasets would fit "
  "on two points. Synthetic data adds datasets whose intrinsic dimension is known."),
 ("Recall measured on distances.", "Duplicate vectors made identifier-based recall wrong, as described above."),
 ("Stronger evaluation.", "The proposal planned one held-out dataset; I used leave-one-dataset-out over all six, "
  "with explicit baselines, so the model has to beat a sensible fixed setting and not just the default."),
 ("New sub-studies.", "Rebuild variance, result-set size and collection size were not in the proposal. Each was "
  "added because a result raised the question."),
]

T["status"] = (
 "At the midpoint the measurement side of the project is complete: the harness is validated, 4,101 "
 "measurements cover twelve datasets, three index types, three result-set sizes and collections from 50,000 to "
 "1,183,514 vectors, and every table and figure regenerates from committed data. The main findings are in "
 "place. IVF-Flat defaults lose recall that a tuned index recovers at no extra latency on every dataset tested. "
 "HNSW defaults lose more as LID and collection size grow. A model built on LID, now with size, predicts the "
 "needed setting within a factor of about 1.5 on datasets it has not seen and, with a margin, reaches the "
 "target on five of six at less than a third of the safe constant's cost. Still incomplete: the GloVe misses are "
 "unexplained, the model predicts only query settings and not build settings, oracle settings have no error "
 "bars yet, and the final report and presentation are not written.")

# ---- condensed for the five-page version ------------------------------------
T["phases"][2] = ("Weeks 5 and 6: more data and the prediction model",
 "I added Fashion-MNIST, MNIST and GloVe-25, plus six synthetic datasets with a known intrinsic dimension "
 "(4 to 48), which also let me check the dimensionality estimator: it reads 4.3 for a true 4 and 7.6 for a "
 "true 8, but only 22.8 for a true 48. Fine-grained sweeps found, for each dataset, the cheapest setting that "
 "reaches a target recall. I built the model (log2 of the needed setting as a linear function of local "
 "intrinsic dimensionality, LID) and tested it leave-one-dataset-out against the library default, fixed "
 "settings and the oracle. I also ran the defaults at k = 1 and k = 100 and on the full million-vector "
 "collections, added 95% confidence intervals, and measured how much recall changes when the same index is "
 "rebuilt. This phase added 2,556 measurements.")
T["heldout_text"] = ("Each dataset is held out in turn: the model is fitted on the other eleven (every size of "
 "the held-out dataset removed) and must predict the held-out dataset's setting. In weeks 5 and 6 the library "
 "default never reached recall 0.95 on a held-out dataset, and the only fixed setting that always did cost 4.79 "
 "times the oracle's latency on HNSW. Week 7 asked whether collection size and hubness improve the model.")
T["size_text"] = ("On HNSW, size matters on its own (Figure 3). From 50,000 vectors to the full collection, the "
 "default's recall falls from 0.805 to 0.689 on SIFT and from 0.562 to 0.464 on GloVe-100, and the ef needed for "
 "0.95 doubles on SIFT (32 to 64) and quadruples on GloVe-100 (256 to 1,024). The fitted size term says the "
 "needed ef grows by a factor of 1.31 each time the collection doubles. Adding it cuts the held-out HNSW error "
 "from 0.79 to 0.58 log2 units and the margin version's cost from 1.55 to 1.30 times the oracle at the same "
 "five-of-six hit rate (Table 1). It also fixes the MNIST over-prediction: the predicted ef drops from 24 to 16, "
 "exactly the true value. For IVF-Flat with a fixed list count the size term is near zero (-0.08) and does not "
 "help. Hubness confirms GloVe-100 is unusual (skew 5.25, against at most 2.84 elsewhere), but adding it barely "
 "moves the error (0.58 to 0.55) and GloVe-100 is still under-predicted (256 against a true 512).")
T["changes"] = [
 ("Baseline check.", "The proposal planned to reproduce a published recall-latency curve. Latency depends on "
  "the hardware, so I used the published ground truth instead, which tests correctness exactly on every dataset."),
 ("Twelve datasets instead of three.", "A held-out test on three datasets would fit a line through two points. "
  "Synthetic data adds datasets whose intrinsic dimension is known."),
 ("Stronger evaluation and new sub-studies.", "Leave-one-dataset-out with explicit baselines replaced a single "
  "held-out dataset. Rebuild variance, result-set size and collection size were added because results raised them."),
]
T["status"] = ("The measurement side is complete: the harness is validated, 4,101 measurements cover twelve "
 "datasets, three index types, three result-set sizes and collections from 50,000 to 1,183,514 vectors, and every "
 "table and figure regenerates from committed data. IVF-Flat defaults lose recall that a tuned index recovers at "
 "no extra latency on every dataset tested; HNSW defaults lose more as LID and collection size grow. The LID and "
 "size model predicts the needed setting within a factor of about 1.5 on unseen datasets and, with a margin, "
 "reaches the target on five of six at 1.30 times the oracle's latency. Still incomplete: the GloVe misses are "
 "unexplained, the model covers query settings but not build settings, oracle settings have no error bars yet, "
 "and the final report and presentation are not written.")

T["overview"][2] = ("Goals.", "Measure how much recall library defaults give up across index types and datasets; "
 "map the best recall achievable at each latency; build and test a model that predicts the setting a dataset "
 "needs from cheap properties of the data instead of an exhaustive sweep; and make it all reproducible.")
T["phases"][1] = ("Weeks 3 and 4: measurement harness and first study",
 "I built the harness in Python with FAISS (IVF-Flat, IVF-PQ), hnswlib (HNSW), NumPy, scikit-learn and pandas: "
 "data loading, exact ground truth, index wrappers, dataset characterization, parameter sweeps and "
 "Pareto-frontier analysis. Checking my exact search against the published neighbors gave 0.9991 agreement on "
 "SIFT but only 0.9856 on NYTimes; all 102 NYTimes disagreements were exact distance ties between duplicate "
 "vectors, so I defined recall on distances, which brought agreement to 1.0000 everywhere. I then ran 657 timed "
 "measurements over 219 configurations on SIFT, GloVe-100 and NYTimes. The nine library defaults averaged a "
 "recall@10 of 0.455.")
T["plan_compare"] = T["plan_compare"].replace("Table 3", "Table 2")
T["status"] = ("The measurement side is complete: 4,101 validated measurements cover twelve datasets, three "
 "index types, three result-set sizes and collections from 50,000 to 1,183,514 vectors, and every table and "
 "figure regenerates from committed data. The LID and size model predicts the needed setting within a factor of "
 "about 1.5 on unseen datasets and, with a margin, reaches the target on five of six at 1.30 times the oracle's "
 "latency. Still incomplete: the GloVe misses are unexplained, the model covers query settings but not build "
 "settings, oracle settings have no error bars yet, and the final report and presentation are not written.")

json.dump(T, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "midterm_text.json"), "w"), indent=1)
print("ok")
