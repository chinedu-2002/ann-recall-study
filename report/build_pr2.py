"""Builds ProgressReport2_Enenta.pdf. Every table is read from results/, so the
report cannot drift from the data. Fonts: Source Serif 4 and Inter, converted
from the @fontsource npm packages (see README)."""
import os, json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                HRFlowable, KeepTogether, PageBreak)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "ProgressReport2_Enenta.pdf")
FONT_DIR = os.environ.get("REPORT_FONT_DIR", "/tmp/f")

for name, fn in [("SS", "SourceSerif4-Regular"), ("SS-B", "SourceSerif4-Semibold"),
                 ("SS-I", "SourceSerif4-Italic"), ("IN", "Inter-Regular"), ("IN-B", "Inter-Semibold")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, fn + ".ttf")))
pdfmetrics.registerFontFamily("SS", normal="SS", bold="SS-B", italic="SS-I", boldItalic="SS-B")

INK, INK2, MUTED = colors.HexColor("#141413"), colors.HexColor("#5c5b57"), colors.HexColor("#8f8e88")
RULE, ZEBRA = colors.HexColor("#d6d5cf"), colors.HexColor("#f7f7f5")

TITLE = ParagraphStyle("T", fontName="IN-B", fontSize=17, leading=21, textColor=INK, spaceAfter=7)
SUB = ParagraphStyle("S", fontName="SS", fontSize=11.5, leading=15.5, textColor=INK2, spaceAfter=10)
BY = ParagraphStyle("BY", fontName="IN", fontSize=9, leading=12, textColor=MUTED, spaceAfter=2)
H2 = ParagraphStyle("H2", fontName="IN-B", fontSize=11.5, leading=14.5, textColor=INK, spaceBefore=16, spaceAfter=7)
H3 = ParagraphStyle("H3", fontName="IN-B", fontSize=9.6, leading=12.5, textColor=INK2, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("B", fontName="SS", fontSize=10.2, leading=14.7, textColor=INK, alignment=TA_JUSTIFY,
                   spaceAfter=8.5, hyphenationLang="en_US", embeddedHyphenation=1)
LEAD = ParagraphStyle("L", parent=B, fontSize=10.7, leading=16)
CAP = ParagraphStyle("C", fontName="SS-I", fontSize=8.4, leading=11.6, textColor=MUTED,
                     alignment=TA_CENTER, spaceBefore=5, spaceAfter=12)
MONO = ParagraphStyle("M", fontName="IN", fontSize=8.2, leading=13, textColor=INK2, leftIndent=12, spaceAfter=9)


def P(t, st=B):
    return Paragraph(t, st)


def tbl(data, widths, size=8.2):
    hs = ParagraphStyle("h", fontName="IN-B", fontSize=size - 0.5, leading=size + 3, textColor=INK2)
    bl = ParagraphStyle("bl", fontName="SS", fontSize=size, leading=size + 3.6, textColor=INK)
    bc = ParagraphStyle("bc", parent=bl, alignment=TA_CENTER)
    rows = [[Paragraph(c, hs) for c in data[0]]]
    rows += [[Paragraph(str(c), bl if i == 0 else bc) for i, c in enumerate(r)] for r in data[1:]]
    t = Table(rows, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.7, INK2), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    return t


def sec(head, body):
    return KeepTogether([P(head, H3), P(body)])


def fig(name, w, h, caption):
    return KeepTogether([Image(os.path.join(FIG, name), width=w * inch, height=h * inch), P(caption, CAP)])


def f3(x):
    return f"{x:.3f}"


SHORT = {"sift-128-euclidean": "SIFT", "glove-100-angular": "GloVe-100", "nytimes-256-angular": "NYTimes",
         "fashion-mnist-784-euclidean": "Fashion-MNIST", "mnist-784-euclidean": "MNIST",
         "glove-25-angular": "GloVe-25"}
REAL = list(SHORT)

# ------------------------------------------------------------------ data
chars = pd.DataFrame(json.load(open(os.path.join(RES, "dataset_characteristics.json")))).set_index("dataset")
gap = pd.read_csv(os.path.join(RES, "pr2_default_gap_6datasets.csv"))
ci_old = pd.DataFrame(json.load(open(os.path.join(RES, "pr1_default_ci.json"))))
summ = pd.read_csv(os.path.join(RES, "pr2_heldout_summary.csv"))
kp = pd.read_csv(os.path.join(RES, "pr2_kpass_default.csv"))
full = pd.read_csv(os.path.join(RES, "pr2_default_gap_fullsize.csv"))
bv = pd.DataFrame(json.load(open(os.path.join(RES, "build_variance.json"))))
nums = json.load(open(os.path.join(RES, "pr2_report_numbers.json")))
EXTRA = json.load(open(os.path.join(ROOT, "report", "pr2_text_numbers.json")))


def default_ci(ds, fam):
    r = gap[(gap.dataset == ds) & (gap.family == fam)]
    if len(r) and pd.notna(r["default_recall_ci95"].iloc[0]):
        return float(r["default_recall"].iloc[0]), float(r["default_recall_ci95"].iloc[0])
    o = ci_old[(ci_old.dataset == ds) & (ci_old.family == fam)]
    return float(r["default_recall"].iloc[0]), float(o["ci95"].iloc[0])


st = []
st += [P("Progress Report 2", TITLE),
       P("Measuring the Recall Cost of Approximate Nearest Neighbor Search: "
         "A Parameter Selection Heuristic for Vector Databases", SUB),
       P("EMMANUEL C. ENENTA", BY), P("Senior Seminar &nbsp;·&nbsp; Fisk University &nbsp;·&nbsp; Weeks 5 and 6", BY),
       HRFlowable(width="100%", thickness=1.0, color=INK2, spaceBefore=12, spaceAfter=4)]

# 1 -------------------------------------------------------------------------
st.append(P("1.&nbsp;&nbsp;Project Overview", H2))
st.append(P(EXTRA["overview"], LEAD))

# 2 -------------------------------------------------------------------------
st.append(P("2.&nbsp;&nbsp;Work Completed", H2))
st.append(P(EXTRA["work_intro"]))
for head, body in EXTRA["work_items"]:
    st.append(sec(head, body))

roster = [["dataset", "vectors used", "dim", "metric", "intrinsic dim (MLE)", "LID at queries"]]
for d in REAL:
    c = chars.loc[d]
    roster.append([SHORT[d], f"{int(c.n_base):,}", int(c.ambient_dim), c.metric,
                   f"{c.intrinsic_dim_mle:.1f}", f"{c.lid_at_queries_mean:.1f}"])
st.append(KeepTogether([tbl(roster, [1.35 * inch, 1.0 * inch, 0.55 * inch, 0.85 * inch, 1.25 * inch, 1.1 * inch]),
                        P("Table 1.  The six real datasets, ordered as in the project, with the dataset "
                          "properties used as heuristic features.", CAP)]))

syn = [["true intrinsic dim", "4", "8", "16", "24", "32", "48"],
       ["MLE estimate"] + [f"{chars.loc[f'synthetic-id{d}-128'].intrinsic_dim_mle:.1f}" for d in (4, 8, 16, 24, 32, 48)],
       ["LID at queries"] + [f"{chars.loc[f'synthetic-id{d}-128'].lid_at_queries_mean:.1f}" for d in (4, 8, 16, 24, 32, 48)]]
st.append(KeepTogether([tbl(syn, [1.4 * inch] + [0.72 * inch] * 6),
                        P("Table 2.  Estimator check on the six synthetic datasets, where the true "
                          "intrinsic dimension is known.", CAP)]))

# 3 -------------------------------------------------------------------------
st.append(P("3.&nbsp;&nbsp;Evidence of Progress and Results", H2))
st.append(sec("3.1&nbsp;&nbsp;The default gap on six datasets, with confidence intervals", EXTRA["gap_text"]))
g = [["dataset (LID)", "HNSW default<br/>recall@10", "IVF-Flat default<br/>recall@10",
      "IVF-Flat best at<br/>same latency", "free gain"]]
for d in sorted(REAL, key=lambda x: chars.loc[x].lid_at_queries_mean):
    h, hc = default_ci(d, "HNSW"); v, vc = default_ci(d, "IVF-Flat")
    best = float(gap[(gap.dataset == d) & (gap.family == "IVF-Flat")]["best_recall_at_default_latency"].iloc[0])
    g.append([f"{SHORT[d]} ({chars.loc[d].lid_at_queries_mean:.0f})", f"{h:.3f} ± {hc:.3f}",
              f"{v:.3f} ± {vc:.3f}", f"{best:.3f}", f"+{best - v:.3f}"])
st.append(KeepTogether([tbl(g, [1.45 * inch, 1.2 * inch, 1.25 * inch, 1.2 * inch, 0.8 * inch]),
                        P("Table 3.  Library defaults on all six datasets at 200,000 base vectors (or the full "
                          "set when smaller), k = 10, sorted by LID. Intervals are 95% over the 300 queries.", CAP)]))
st.append(fig("fig2_default_gap_six.png", 6.35, 2.9,
              "Figure 1.  Default recall (gray) against the best measured configuration at the same latency "
              "(colored). Rows are sorted by LID; the number in parentheses is the LID."))

st.append(sec("3.2&nbsp;&nbsp;The heuristic: LID predicts the setting a dataset needs", EXTRA["fit_text"]))
st.append(fig("fig4_heuristic_fit.png", 6.35, 3.1,
              "Figure 2.  Smallest query setting reaching recall@10 of 0.95 against query-point LID, for all "
              "twelve datasets. Filled points are real datasets, hollow points synthetic. The dashed line is "
              "the one-feature fit; the vertical axis is logarithmic."))

st.append(sec("3.3&nbsp;&nbsp;Held-out evaluation", EXTRA["heldout_text"]))
LAB = {"heuristic": "LID heuristic", "heuristic+margin": "LID heuristic + margin",
       "constant-max": "safe constant", "constant-median": "median constant", "default": "library default"}
ho = [["method", "HNSW<br/>hit rate", "HNSW min<br/>recall", "HNSW cost<br/>vs oracle",
       "IVF hit<br/>rate", "IVF min<br/>recall", "IVF cost<br/>vs oracle"]]
S = summ[(summ.training == "real+synthetic") & (summ.feature == "lid") & (summ.target == 0.95)]
for m in ["heuristic", "heuristic+margin", "constant-max", "constant-median", "default"]:
    rh = S[(S.family == "HNSW") & (S.method == m)].iloc[0]; ri = S[(S.family == "IVF-Flat") & (S.method == m)].iloc[0]
    ho.append([LAB[m], f"{round(rh.hit_rate * 6)} / 6", f3(rh.min_recall), f"{rh.geo_latency_vs_oracle:.2f}×",
               f"{round(ri.hit_rate * 6)} / 6", f3(ri.min_recall), f"{ri.geo_latency_vs_oracle:.2f}×"])
st.append(KeepTogether([tbl(ho, [1.55 * inch, 0.72 * inch, 0.78 * inch, 0.82 * inch, 0.72 * inch, 0.78 * inch, 0.82 * inch]),
                        P("Table 4.  Leave-one-dataset-out results at a target of recall@10 = 0.95, over the six "
                          "real datasets. Cost is the geometric mean of latency relative to the oracle setting.", CAP)]))
st.append(fig("fig5_heldout_hnsw.png", 6.35, 3.6,
              "Figure 3.  HNSW held-out detail. Left: recall reached on each held-out dataset. Right: latency "
              "relative to the oracle setting for that dataset (the default is omitted on the right because it "
              "never reaches the target)."))

st.append(sec("3.4&nbsp;&nbsp;Which feature carries the signal", EXTRA["ablation_text"]))
ab = [["feature used by the model", "HNSW mean |log2 error|", "IVF-Flat mean |log2 error|"]]
A = summ[(summ.training == "real+synthetic") & (summ.method == "heuristic") & (summ.target == 0.95)]
for f, lab in (("lid", "LID at queries"), ("intrinsic_dim", "intrinsic dim (MLE)"),
               ("ambient_dim", "ambient dimension"), ("none", "none (constant)")):
    ab.append([lab, f"{A[(A.family == 'HNSW') & (A.feature == f)].mean_abs_log2_error.iloc[0]:.2f}",
               f"{A[(A.family == 'IVF-Flat') & (A.feature == f)].mean_abs_log2_error.iloc[0]:.2f}"])
st.append(KeepTogether([tbl(ab, [2.2 * inch, 1.8 * inch, 1.8 * inch]),
                        P("Table 5.  Feature ablation, target 0.95. An error of 1.0 means the predicted setting "
                          "was off by a factor of two on average.", CAP)]))

st.append(sec("3.5&nbsp;&nbsp;Result-set size and the ef clamp", EXTRA["kpass_text"]))
st.append(fig("fig6_kpass.png", 6.35, 2.6,
              "Figure 4.  HNSW recall against ef at k = 1, 10 and 100, default build. Rings mark the default "
              "ef of 10. The flat orange segment from ef 10 to ef 100 is the clamp."))

st.append(sec("3.6&nbsp;&nbsp;Full-size base sets", EXTRA["full_text"]))
fs = [["dataset", "family", "default recall<br/>200k vectors", "default recall<br/>full set", "best at same<br/>latency, full set"]]
for d in ("sift-128-euclidean", "glove-100-angular"):
    for fam in ("HNSW", "IVF-Flat"):
        small, _ = default_ci(d, fam)
        r = full[(full.dataset == d) & (full.family == fam)].iloc[0]
        fs.append([f"{SHORT[d]} ({int(r.n_base):,})", fam, f3(small),
                   f"{r.default_recall:.3f} ± {r.default_recall_ci95:.3f}", f3(r.best_recall_at_default_latency)])
st.append(KeepTogether([tbl(fs, [1.7 * inch, 0.8 * inch, 1.1 * inch, 1.25 * inch, 1.2 * inch]),
                        P("Table 6.  Default recall at 200,000 vectors against the full collections.", CAP)]))

st.append(sec("3.7&nbsp;&nbsp;Code and reproducibility", EXTRA["code_text"]))
st.append(P("src/heuristic.py &nbsp;&nbsp; src/sweep.py (profiles pr1, dense, kpass, scale)<br/>"
            "scripts/make_synthetic.py &nbsp;&nbsp; scripts/run_pr2_experiments.sh &nbsp;&nbsp; scripts/analyze_pr2.py<br/>"
            "scripts/build_variance.py &nbsp;&nbsp; scripts/default_ci.py<br/>"
            "scripts/live_demo.py &nbsp;&nbsp; scripts/predict_demo.py", MONO))
st.append(P('Repository: <link href="https://github.com/chinedu-2002/ann-recall-study" color="#2a78d6">'
            'github.com/chinedu-2002/ann-recall-study</link>'))

# 4 -------------------------------------------------------------------------
st.append(P("4.&nbsp;&nbsp;Challenges", H2))
for head, body in EXTRA["challenges"]:
    st.append(P(f"<b>{head}</b> {body}"))

# 5 -------------------------------------------------------------------------
st.append(P("5.&nbsp;&nbsp;Next Steps", H2))
st.append(P(EXTRA["next"]))


def footer(c, d):
    c.saveState(); c.setFont("IN", 7.5); c.setFillColor(MUTED)
    c.drawString(1.0 * inch, 0.55 * inch, "Enenta  ·  Progress Report 2")
    c.drawRightString(letter[0] - 1.0 * inch, 0.55 * inch, str(d.page))
    c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(1.0 * inch, 0.72 * inch, letter[0] - 1.0 * inch, 0.72 * inch)
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.85 * inch,
                        bottomMargin=0.85 * inch, title="Progress Report 2", author="Emmanuel C. Enenta")
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("built", OUT)
