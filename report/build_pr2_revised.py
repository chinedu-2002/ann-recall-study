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
                                HRFlowable, KeepTogether, PageBreak, CondPageBreak)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "ProgressReport2_Enenta_revised.pdf")  # short version + theory
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
H2 = ParagraphStyle("H2", fontName="IN-B", fontSize=11.5, leading=14.5, textColor=INK, spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", fontName="IN-B", fontSize=9.6, leading=12.5, textColor=INK2, spaceBefore=9, spaceAfter=4)
B = ParagraphStyle("B", fontName="SS", fontSize=10.1, leading=14.1, textColor=INK, alignment=TA_JUSTIFY,
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
EXTRA = json.load(open(os.path.join(ROOT, "report", "pr2_short_text.json")))
THEORY = [t for t in json.load(open(os.path.join(ROOT, "report", "midterm_text.json")))["theory"] if t[0] != "Hubness."]
EXTRA["fit_text"] = EXTRA["fit_text"].replace("Figure 1", "Figure 2")


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
       HRFlowable(width="100%", thickness=1.0, color=INK2, spaceBefore=10, spaceAfter=2)]

st.extend(h2("1.&nbsp;&nbsp;Project Overview"))
st.append(P(EXTRA["overview"], LEAD))


st.append(P("1.1&nbsp;&nbsp;How the methods work", H3))
for head, body in THEORY:
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

st.extend(h2("2.&nbsp;&nbsp;Work Completed"))
st.append(P(EXTRA["work_intro"]))
for head, body in EXTRA["work_items"]:
    st.append(P(f"<b>{head}</b> {body}"))
st.append(P('<b>Repository.</b> Code, raw results and both demos: '
            '<link href="https://github.com/chinedu-2002/ann-recall-study" color="#2a78d6">'
            'github.com/chinedu-2002/ann-recall-study</link>'))

st += [CondPageBreak(1.3 * inch), P("3.&nbsp;&nbsp;Evidence of Progress and Results", H2),
       P("3.1&nbsp;&nbsp;The default gap on six datasets", H3), P(EXTRA["gap_text"])]
g = [["dataset (LID)", "HNSW default<br/>recall@10", "IVF-Flat default<br/>recall@10",
      "IVF-Flat best at<br/>same latency", "free gain"]]
for d in sorted(REAL, key=lambda x: chars.loc[x].lid_at_queries_mean):
    h, hc = default_ci(d, "HNSW"); v, vc = default_ci(d, "IVF-Flat")
    best = float(gap[(gap.dataset == d) & (gap.family == "IVF-Flat")]["best_recall_at_default_latency"].iloc[0])
    g.append([f"{SHORT[d]} ({chars.loc[d].lid_at_queries_mean:.0f})", f"{h:.3f} ± {hc:.3f}",
              f"{v:.3f} ± {vc:.3f}", f"{best:.3f}", f"+{best - v:.3f}"])
st.append(KeepTogether([tbl(g, [1.45 * inch, 1.2 * inch, 1.25 * inch, 1.2 * inch, 0.8 * inch]),
                        P("Table 2.  Library defaults at k = 10, sorted by LID. Intervals are 95% over 300 queries.", CAP)]))

st.extend(sec("3.2&nbsp;&nbsp;LID predicts the setting a dataset needs", EXTRA["fit_text"]))
st.append(fig("fig4_heuristic_fit.png", 5.3, 2.59,
              "Figure 2.  Smallest setting reaching recall@10 of 0.95 against LID, twelve datasets. Filled points "
              "are real, hollow points synthetic; the dashed line is the fitted model (log scale)."))

st.extend(sec("3.3&nbsp;&nbsp;Testing the model on datasets it has not seen", EXTRA["heldout_text"]))
LAB = {"heuristic": "LID model", "heuristic+margin": "LID model + margin",
       "constant-max": "safe constant", "constant-median": "median constant", "default": "library default"}
ho = [["method", "HNSW<br/>hits", "HNSW cost<br/>vs oracle", "IVF-Flat<br/>hits", "IVF-Flat cost<br/>vs oracle"]]
S = summ[(summ.training == "real+synthetic") & (summ.feature == "lid") & (summ.target == 0.95)]
for m in ["heuristic", "heuristic+margin", "constant-max", "constant-median", "default"]:
    rh = S[(S.family == "HNSW") & (S.method == m)].iloc[0]; ri = S[(S.family == "IVF-Flat") & (S.method == m)].iloc[0]
    ho.append([LAB[m], f"{round(rh.hit_rate * 6)} / 6", f"{rh.geo_latency_vs_oracle:.2f}×",
               f"{round(ri.hit_rate * 6)} / 6", f"{ri.geo_latency_vs_oracle:.2f}×"])
st.append(KeepTogether([tbl(ho, [1.7 * inch, 0.9 * inch, 1.1 * inch, 0.9 * inch, 1.2 * inch]),
                        P("Table 3.  Leave-one-dataset-out on the six real datasets, target recall@10 = 0.95.", CAP)]))

st.extend(sec("3.4&nbsp;&nbsp;Other checks", EXTRA["other_text"]))

st.extend(h2("4.&nbsp;&nbsp;Challenges"))
for head, body in EXTRA["challenges"]:
    st.append(P(f"<b>{head}</b> {body}"))

st.extend(h2("5.&nbsp;&nbsp;Next Steps"))
st.append(P(EXTRA["next"]))


def footer(c, d):
    c.saveState(); c.setFont("IN", 7.5); c.setFillColor(MUTED)
    c.drawString(1.0 * inch, 0.55 * inch, "Enenta  ·  Progress Report 2")
    c.drawRightString(letter[0] - 1.0 * inch, 0.55 * inch, str(d.page))
    c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(1.0 * inch, 0.72 * inch, letter[0] - 1.0 * inch, 0.72 * inch)
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.8 * inch,
                        bottomMargin=0.85 * inch, title="Progress Report 2", author="Emmanuel C. Enenta")
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("built", OUT)
