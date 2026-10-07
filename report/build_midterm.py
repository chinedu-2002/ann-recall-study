"""Builds MidtermReport_Enenta.pdf. Every table is read from results/, so the
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
                                HRFlowable, KeepTogether, PageBreak, CondPageBreak)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "MidtermReport_Enenta.pdf")
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
H2 = ParagraphStyle("H2", fontName="IN-B", fontSize=11.5, leading=14.5, textColor=INK, spaceBefore=12, spaceAfter=5)
H3 = ParagraphStyle("H3", fontName="IN-B", fontSize=9.6, leading=12.5, textColor=INK2, spaceBefore=9, spaceAfter=4)
B = ParagraphStyle("B", fontName="SS", fontSize=10.0, leading=13.7, textColor=INK, alignment=TA_JUSTIFY,
                   spaceAfter=7, hyphenationLang="en_US", embeddedHyphenation=1)
LEAD = ParagraphStyle("L", parent=B, fontSize=10.7, leading=16)
CAP = ParagraphStyle("C", fontName="SS-I", fontSize=8.4, leading=11.6, textColor=MUTED,
                     alignment=TA_CENTER, spaceBefore=4, spaceAfter=9)
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
    # break before a heading only when fewer than ~4 lines would fit under it
    return [CondPageBreak(0.95 * inch), P(head, H3), P(body)]


def h2(text):
    return [CondPageBreak(0.85 * inch), P(text, H2)]


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
EXTRA = json.load(open(os.path.join(ROOT, "report", "midterm_text.json")))
w7 = pd.read_csv(os.path.join(RES, "week7_heldout_summary.csv"))


def default_ci(ds, fam):
    r = gap[(gap.dataset == ds) & (gap.family == fam)]
    if len(r) and pd.notna(r["default_recall_ci95"].iloc[0]):
        return float(r["default_recall"].iloc[0]), float(r["default_recall_ci95"].iloc[0])
    o = ci_old[(ci_old.dataset == ds) & (ci_old.family == fam)]
    return float(r["default_recall"].iloc[0]), float(o["ci95"].iloc[0])




from reportlab.platypus import Preformatted
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=7.9, leading=10.6, textColor=INK, leftIndent=14,
                      backColor=colors.HexColor("#f4f4f1"), borderPadding=(6, 8, 6, 8), spaceBefore=4, spaceAfter=10)
CELL = ParagraphStyle("cell", fontName="SS", fontSize=8.0, leading=10.8, textColor=INK)


def ltbl(data, widths, size=8.0):
    """Table whose cells are all left-aligned paragraphs, for text-heavy rows."""
    hs = ParagraphStyle("h", fontName="IN-B", fontSize=size - 0.5, leading=size + 3, textColor=INK2)
    cs = ParagraphStyle("c", parent=CELL, fontSize=size, leading=size + 2.8)
    rows = [[Paragraph(c, hs) for c in data[0]]] + [[Paragraph(str(c), cs) for c in r] for r in data[1:]]
    t = Table(rows, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.7, INK2), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    return t


st = []
st += [P("Midterm Project Report", TITLE),
       P("Measuring the Recall Cost of Approximate Nearest Neighbor Search: "
         "A Parameter Selection Heuristic for Vector Databases", SUB),
       P("EMMANUEL C. ENENTA", BY), P("Senior Seminar &nbsp;·&nbsp; Fisk University &nbsp;·&nbsp; Weeks 1 to 7", BY),
       HRFlowable(width="100%", thickness=1.0, color=INK2, spaceBefore=10, spaceAfter=2)]

# 1 ---------------------------------------------------------------------------
st.extend(h2("1.&nbsp;&nbsp;Project Overview"))
for head, body in EXTRA["overview"]:
    st.append(P(f"<b>{head}</b> {body}"))


st.append(P("1.1&nbsp;&nbsp;How the methods work", H3))
for head, body in EXTRA["theory"]:
    st.append(P(f"<b>{head}</b> {body}"))
    if head.startswith("HNSW"):
        st.append(fig("fig0_methods.png", 6.5, 2.38,
                      "Figure 1.  How each index answers a query (toy 2-D data, for illustration). Left: IVF scans "
                      "only the nprobe cells nearest the query, so a neighbor just across a border is missed at "
                      "nprobe = 1. Right: HNSW walks greedily down a stack of graphs, then keeps ef candidates on "
                      "the bottom layer."))
    if head.startswith("The trade-off"):
        dz = pd.read_csv(os.path.join(RES, "week7_dense_aggregated.csv"))
        dz = dz[(dz.dataset == "sift-128-euclidean") & (dz.n_base == 200000)]
        tt = [["index", "setting", "recall@10", "ms per query", "note"]]
        for fam, name, vals, notes in [("HNSW", "ef", [10, 48, 1024], {10: "library default", 48: "oracle for 0.95"}),
                                       ("IVF-Flat", "nprobe", [1, 32, 256], {1: "library default", 32: "oracle for 0.95"})]:
            for v in vals:
                r = dz[(dz.family == fam) & (dz.query_param == v)].iloc[0]
                tt.append([fam, f"{name} = {v}", f"{r.recall:.3f}", f"{r.latency_ms:.3f}", notes.get(v, "")])
        st.append(KeepTogether([tbl(tt, [1.0 * inch, 1.15 * inch, 0.95 * inch, 1.1 * inch, 1.4 * inch]),
                                P("Table 1.  The speed and accuracy trade-off on SIFT (200,000 vectors, IVF with 1,024 "
                                  "cells, one search thread).", CAP)]))

# 2 ---------------------------------------------------------------------------
st.extend(h2("2.&nbsp;&nbsp;Work Completed: Weeks 1 to 7"))
st.append(P(EXTRA["work_intro"]))
for head, body in EXTRA["phases"]:
    st.extend(sec(head, body))

# 3 ---------------------------------------------------------------------------
st.extend(h2("3.&nbsp;&nbsp;Evidence of Progress"))
st.append(P('<b>Repository.</b> All code, raw measurements, figures and both demos: '
            '<link href="https://github.com/chinedu-2002/ann-recall-study" color="#2a78d6">'
            'github.com/chinedu-2002/ann-recall-study</link>. Running <font face="Courier">python scripts/predict_demo.py</font> '
            'generates an unseen dataset, predicts its setting and checks the prediction live.', ParagraphStyle("BL", parent=B, alignment=0)))
st.extend(sec("3.1&nbsp;&nbsp;The default gap on six datasets", EXTRA["gap_text"]))
st.append(fig("fig2_default_gap_six.png", 4.9, 2.23,
              "Figure 2.  Default recall (gray) against the best measured configuration at the same latency "
              "(colored), k = 10. Rows are sorted by LID, shown in parentheses."))
st.extend(sec("3.2&nbsp;&nbsp;The prediction model", EXTRA["fit_text"]))
st.append(fig("fig4_heuristic_fit.png", 4.1, 2.0,
              "Figure 3.  Smallest setting reaching recall@10 of 0.95 against LID for all twelve datasets. "
              "Filled points are real, hollow points synthetic; the dashed line is the fitted model."))
st.extend(sec("3.3&nbsp;&nbsp;Held-out evaluation, collection size and hubness", EXTRA["heldout_text"]))
st.append(P(EXTRA["size_text"]))
st.append(fig("fig7_size.png", 4.6, 1.96,
              "Figure 4.  SIFT and GloVe-100 at five collection sizes. Left: HNSW recall at the default ef. "
              "Right: the smallest ef reaching recall 0.95 (log scale)."))
w = w7[(w7.test == "primary") & (w7.target == 0.95)]
t2 = [["model features", "HNSW error<br/>(log2)", "HNSW + margin:<br/>hits, cost", "IVF-Flat error<br/>(log2)",
       "IVF-Flat + margin:<br/>hits, cost"]]
for m in ["none", "LID", "LID + size", "LID + size + hubness"]:
    def g(fam, meth, col):
        return w[(w.family == fam) & (w.model == m) & (w.method == meth)][col].iloc[0]
    t2.append(["none (one fixed setting)" if m == "none" else m, f'{g("HNSW", "model", "err"):.2f}',
               f'{int(g("HNSW", "model + margin", "hits"))} / 6, {g("HNSW", "model + margin", "cost"):.2f}×',
               f'{g("IVF-Flat", "model", "err"):.2f}',
               f'{int(g("IVF-Flat", "model + margin", "hits"))} / 6, {g("IVF-Flat", "model + margin", "cost"):.2f}×'])
st.append(KeepTogether([tbl(t2, [1.55 * inch, 0.95 * inch, 1.25 * inch, 1.0 * inch, 1.25 * inch]),
                        P("Table 2.  Held-out test, target recall@10 = 0.95. Error is the mean distance between the "
                          "predicted and true setting in log2 units; 0.58 means within a factor of about 1.5.", CAP)]))

st.append(P("3.4&nbsp;&nbsp;Code example", H3))
st.append(P("The core metric, tie-aware recall, counts a returned neighbor as correct if it is at least as close "
            "as the true k-th neighbor, so duplicate vectors are not penalized (src/sweep.py):"))
st.append(Preformatted(
    "def per_query_tieaware(found_d, truth_d, k, rtol=1e-5):\n"
    "    thresh = truth_d[:, k - 1:k]                     # distance of the true k-th neighbor\n"
    "    tol = rtol * np.maximum(1.0, np.abs(thresh))     # floating-point tolerance\n"
    "    return (found_d[:, :k] <= thresh + tol).sum(axis=1) / k", CODE))

# 4 ---------------------------------------------------------------------------
st.extend(h2("4.&nbsp;&nbsp;Progress Compared with the Original Proposal"))
st.append(P(EXTRA["plan_compare"]))
t3 = [["planned task", "planned", "actual status"],
      ["Measurement harness and baseline check", "weeks 1 to 3",
       "<b>Done</b> in weeks 3 and 4. Baseline check changed to published ground truth (see below)."],
      ["Parameter sweeps, three index types on three datasets", "weeks 4 to 8",
       "<b>Done and extended</b> by week 7: six real and six synthetic datasets, k = 1 and 100, full collections, "
       "five collection sizes."],
      ["Fit and validate the heuristic on held-out data", "weeks 9 to 11",
       "<b>First version done</b> in weeks 5 and 6, size and hubness added in week 7. <b>In progress:</b> "
       "build parameters, error bars."],
      ["Analysis, final report, presentation", "weeks 12 to 14",
       "<b>Not started</b>, but the analysis and report pipeline already exists from the progress reports."]]
st.append(KeepTogether([ltbl(t3, [2.2 * inch, 0.95 * inch, 3.35 * inch]),
                        P("Table 3.  Original timeline against actual progress.", CAP)]))
st.append(P("<b>Changes to the plan and why.</b>"))
for head, body in EXTRA["changes"]:
    st.append(P(f"<i>{head}</i> {body}"))

# 5 ---------------------------------------------------------------------------
st.extend(h2("5.&nbsp;&nbsp;Challenges and Solutions"))
t4 = [["challenge", "effect on the project", "what I did", "status"],
      ["Exact search disagreed with published neighbors on NYTimes (0.9856)",
       "Recall would have been understated", "Traced all 102 cases to distance ties; measured recall on distances",
       "Resolved"],
      ["hnswlib silently raises ef to at least k", "First sweep showed a plateau that did not exist",
       "Grid respects the clamp; confirmed directly at k = 100", "Resolved, now a finding"],
      ["Intrinsic-dimension estimator returned 7.5 × 10<super>8</super> on NYTimes",
       "Main heuristic feature unusable", "Drop zero distances from duplicates, trimmed mean; validated on synthetic data",
       "Resolved"],
      ["Rebuilding the same index changes recall (NYTimes 0.590 to 0.679)",
       "One build understates uncertainty", "Measured by thread count and insertion order; ranges reported",
       "Open: error bars planned"],
      ["Collection size tangled up with LID", "MNIST over-predicted", "Week 7 size sweep; size term added",
       "Resolved for HNSW"],
      ["Estimator saturates; synthetic LID tops out near 26", "High end of the model rests on two datasets",
       "Flagged in every report", "Open"]]
st.append(KeepTogether([ltbl(t4, [1.95 * inch, 1.45 * inch, 2.0 * inch, 1.1 * inch], size=7.7),
                        P("Table 4.  Major challenges from weeks 1 to 7.", CAP)]))

# 6 ---------------------------------------------------------------------------
st.extend(h2("6.&nbsp;&nbsp;Current Project Status"))
st.append(P(EXTRA["status"]))

# 7 ---------------------------------------------------------------------------
st.extend(h2("7.&nbsp;&nbsp;Plan for the Second Half of the Semester"))
t5 = [["weeks", "tasks", "milestone"],
      ["8", "Rebuild every oracle setting three times with shuffled insertion order; refit the LID and size model "
            "with those error bars.", "Oracle settings with uncertainty"],
      ["9 and 10", "Extend the model to build settings (HNSW M, IVF list count) and their memory cost; test "
                   "other features for word embeddings; make the synthetic generator reach LID 40 to 55.",
       "Model covers build and query settings"],
      ["11", "Final held-out evaluation, including full-size collections; freeze all results.", "Final results"],
      ["12 and 13", "Write the final report: methods, results, limitations.", "Report draft, then final"],
      ["14", "Final presentation and repository clean-up with a one-command reproduction script.",
       "Presentation delivered"]]
st.append(KeepTogether([ltbl(t5, [0.8 * inch, 4.0 * inch, 1.7 * inch]),
                        P("Table 5.  Plan for weeks 8 to 14.", CAP)]))


def footer(c, d):
    c.saveState(); c.setFont("IN", 7.5); c.setFillColor(MUTED)
    c.drawString(1.0 * inch, 0.55 * inch, "Enenta  ·  Midterm Project Report")
    c.drawRightString(letter[0] - 1.0 * inch, 0.55 * inch, str(d.page))
    c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(1.0 * inch, 0.72 * inch, letter[0] - 1.0 * inch, 0.72 * inch)
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.8 * inch,
                        bottomMargin=0.85 * inch, title="Midterm Project Report", author="Emmanuel C. Enenta")
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("built", OUT)
