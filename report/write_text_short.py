"""Short version of the Progress Report 2 prose (about four pages)."""
import json, os
T = {}
T["overview"] = (
 "This project measures how much recall approximate nearest neighbor (ANN) indexes lose at their library "
 "default settings, and tests whether the setting a dataset needs can be predicted from a cheap statistic of "
 "the data instead of found by an exhaustive sweep. In weeks 5 and 6 I built that prediction model, a step "
 "originally planned for weeks 9 to 11, and tested it on datasets it had not seen. I also doubled the real "
 "datasets from three to six, checked result-set size and full-size collections, and put confidence intervals "
 "on the main numbers. Two of these checks changed conclusions from Progress Report 1.")
T["work_intro"] = (
 "The codebase grew from 889 to 1,701 lines of Python, and this period added 2,556 timed measurements, for a "
 "total of 3,213.")
T["work_items"] = [
 ("More data.", "I added Fashion-MNIST and MNIST (60,000 images, 784 dimensions) and GloVe-25 (25-dimensional "
  "word vectors), plus six synthetic datasets whose intrinsic dimension I control, from 4 to 48. The synthetic "
  "sets let me check the dimensionality estimator against known answers. It reads 4.3 for a true 4 and 7.6 for "
  "a true 8, but only 22.8 for a true 48, so it underestimates as the dimension grows."),
 ("Fine-grained sweeps.", "21 HNSW ef values and 16 IVF nprobe values on all twelve datasets, three timed "
  "repeats each, to find the cheapest setting that reaches a target recall."),
 ("The prediction model.", "log2 of the needed setting is a linear function of one feature, the local "
  "intrinsic dimensionality (LID) measured at the query points. I tested it leave-one-dataset-out on the six "
  "real datasets: fit on everything else, predict the held-out dataset, then check the recall and latency that "
  "prediction actually gets. Baselines are the library default, a safe constant (the largest value that "
  "worked in training), a median constant and the oracle. A second version adds a margin fixed in advance."),
 ("Live demos.", "scripts/predict_demo.py generates a dataset the model has never seen, measures its LID in "
  "under a second, predicts the setting, and checks it on the spot."),
]
T["gap_text"] = (
 "Across all eighteen dataset-and-index combinations the mean default recall@10 is 0.536, ranging from 0.128 "
 "to 0.930. That corrects Progress Report 1, which said no default reached 0.75: on MNIST and Fashion-MNIST the "
 "HNSW default reaches 0.929 and 0.930. HNSW defaults get worse as LID rises, down to 0.515 on GloVe-100. The "
 "IVF-Flat finding holds on every dataset: a tuned index beats the default at the same latency by between 0.171 "
 "and 0.291 recall.")
T["fit_text"] = (
 "Across all twelve datasets, the log of the smallest setting that reaches recall 0.95 tracks query-point LID "
 "with a correlation of 0.963 for HNSW and 0.959 for IVF-Flat (Figure 1). For HNSW, the ef a dataset needs "
 "doubles for every 7.3 points of LID. Synthetic and real datasets fall on the same line.")
T["heldout_text"] = (
 "The library default never reaches 0.95 on a held-out dataset. The safe constant works, but on HNSW it costs "
 "4.79 times the oracle's latency. The bare model hits the target on four of six datasets at 1.17 and 1.00 "
 "times the oracle cost. With the margin it hits five of six for both index types at 1.44 and 1.42 times, about "
 "a third of what the safe constant spends. Both misses are GloVe word embeddings, which reached 0.937 and 0.938. "
 "Swapping LID for plain vector dimension makes the model barely better than guessing a constant (average error "
 "1.68 against 1.97 log2 units, versus 0.72 with LID), so LID is what carries the signal.")
T["other_text"] = (
 "At k = 100, ef = 10 and ef = 100 give identical recall on all three datasets tested, to five decimals, which "
 "confirms that hnswlib silently raises ef to k. At full size the HNSW default gets worse (GloVe-100 falls from "
 "0.515 to 0.478, SIFT to 0.696), while the IVF-Flat gain survives: a tuned index reaches 0.827 on SIFT and "
 "0.810 on GloVe-100 at the default's latency.")
T["challenges"] = [
 ("My headline did not survive more data.", "Two HNSW defaults reach 0.93, so last report's claim that no "
  "default reached 0.75 no longer holds. The IVF-Flat gap appears on all six datasets; the HNSW gap depends on "
  "the dataset."),
 ("Rebuilding the same index changes recall.", "NYTimes measured 0.679 last report and 0.604 this period with "
  "identical settings. Single-threaded builds reproduce 0.679 exactly, but shuffling the insertion order gives "
  "0.643 to 0.652, and two-threaded builds give 0.590 to 0.634. That 0.089 spread is as large as the entire "
  "±0.044 interval from sampling queries, so last report's NYTimes figure was a favorable draw."),
 ("Size and estimator limits.", "The two easiest datasets are also the smallest, so part of the LID effect may "
  "be a size effect. The estimator also saturates, so the synthetic data only reaches an LID of about 26, and "
  "the high end of the model rests on two real datasets."),
]
T["next"] = (
 "In weeks 7 and 8 I will separate collection size from LID with sweeps from 50,000 to 1,000,000 vectors, test "
 "a hubness feature for the GloVe misses, rebuild each oracle three times to give it an error bar, and extend the "
 "model to build settings (HNSW M, IVF nlist).")
json.dump(T, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr2_short_text.json"), "w"), indent=1)
print("ok")
