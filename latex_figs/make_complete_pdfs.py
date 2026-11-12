"""One complete PDF per figure folder (A4 landscape, every figure at 100 % size so labels keep their 10 pt,
a short heading above each figure). Run from latex_figs after the figures are built:
    ~/miniconda3/envs/qchem/bin/python make_complete_pdfs.py"""
import os, re, glob, fitz
HERE = os.path.dirname(os.path.abspath(__file__))
QN = {"": "C_GME", "_purity": "purity", "_linear_entropy": "linear entropy", "_vN_entropy": "von Neumann entropy",
      "_omega": "detection probability omega_n"}

def figures(folder, kind):
    """(path, heading, sort key) for the figure PDFs of one folder; kind = 'paper' or 'nodoublon'."""
    out = []
    for f in glob.glob(os.path.join(HERE, folder, "*.pdf")):
        n = os.path.basename(f)[:-4]
        if kind == "paper":
            m = re.fullmatch(r"fig(\d)(_purity|_linear_entropy|_vN_entropy)?", n)
            if m: out.append((f, f"Fig. {m.group(1)}: {QN[m.group(2) or '']}", (int(m.group(1)), "", list(QN).index(m.group(2) or ""))))
        else:
            m = re.fullmatch(r"fig(\d)([ab]?)_proj(_purity|_linear_entropy|_omega)?", n)
            if m: out.append((f, f"Fig. {m.group(1)}{'(' + m.group(2) + ')' if m.group(2) else ''} without doublons: {QN[m.group(3) or '']}",
                              (int(m.group(1)), m.group(2), list(QN).index(m.group(3) or ""))))
    return [(p, h) for p, h, _ in sorted(out, key=lambda t: t[2])]

def merged(folder):
    f = os.path.join(HERE, folder, "fig12_combined.pdf")
    return [(f, "Fig. 1 + Fig. 2 merged")] if os.path.exists(f) else []

def write(outfile, title, items):
    W, H, M, HEAD, GX, GY, TOP = 842, 595, 14, 16, 8, 10, 30
    doc = fitz.open(); st = {"pg": None}
    def new_page():
        st["pg"] = doc.new_page(width=W, height=H); st.update(x=M, y=TOP, rowh=0)
        st["pg"].insert_text((M, 18), title, fontsize=9.5)
    for path, head in items:
        src = fitz.open(path); r = src[0].rect; h = HEAD + r.height
        assert r.width <= W - 2 * M and h <= H - TOP - M, path
        if st["pg"] is None: new_page()
        if st["x"] + r.width > W - M: st.update(x=M, y=st["y"] + st["rowh"] + GY, rowh=0)
        if st["y"] + h > H - M: new_page()
        st["pg"].insert_text((st["x"], st["y"] + 10), head, fontsize=9)
        st["pg"].show_pdf_page(fitz.Rect(st["x"], st["y"] + HEAD, st["x"] + r.width, st["y"] + HEAD + r.height), src, 0)
        st["x"] += r.width + GX; st["rowh"] = max(st["rowh"], h)
    n = len(doc)
    for i, p in enumerate(doc): p.insert_text((W - M - 60, 18), f"page {i + 1}/{n}", fontsize=9)
    doc.save(os.path.join(HERE, outfile), garbage=4, deflate=True)
    print(f"{outfile}: {n} pages, {len(items)} figures")

JOBS = [
 ("pdf_10pt/all_figures_10pt.pdf", "Paper figures (10 pt labels, 1.2 pt lines)", figures("pdf_10pt", "paper")),
 ("projected/all_figures_without_doublons.pdf", "Figures without doublons", figures("projected", "nodoublon")),
 ("cosmetic/all_figures_cosmetic.pdf", "Paper C_GME figures with curve labels and transition marks", figures("cosmetic", "paper")),
 ("shadow/all_figures_shadow.pdf", "Figures with line shadows",
  figures("shadow/paper", "paper") + merged("shadow/fig12_combined") + figures("shadow/projected", "nodoublon")),
 ("shadow/paper_figures_shadow.pdf", "Paper figures with line shadows", figures("shadow/paper", "paper") + merged("shadow/fig12_combined")),
 ("shadow/without_doublons_figures_shadow.pdf", "Figures without doublons, with line shadows", figures("shadow/projected", "nodoublon")),
 ("shadow_cosmetic/paper_figures_shadow_cosmetic.pdf", "Paper figures with line shadows, transition marks and curve labels",
  figures("shadow_cosmetic/paper", "paper") + merged("shadow_cosmetic/fig12_combined")),
 ("shadow_cosmetic/without_doublons_figures_shadow_cosmetic.pdf", "Figures without doublons, with line shadows and transition marks",
  figures("shadow_cosmetic/projected", "nodoublon")),
]
def cgme_only(items):
    """keep only the C_GME figures (no purity, entropy or omega_n)"""
    return [(p, h) for p, h in items if h.endswith("C_GME") or h.startswith("Fig. 1 + Fig. 2")]

JOBS += [
 ("shadow/cgme_only_shadow.pdf", "C_GME figures with line shadows",
  cgme_only(figures("shadow/paper", "paper") + merged("shadow/fig12_combined") + figures("shadow/projected", "nodoublon"))),
 ("shadow_cosmetic/cgme_only_shadow_cosmetic.pdf", "C_GME figures with line shadows, transition marks and curve labels",
  cgme_only(figures("shadow_cosmetic/paper", "paper") + merged("shadow_cosmetic/fig12_combined") + figures("shadow_cosmetic/projected", "nodoublon"))),
]
JOBS += [
 ("shadow_cosmetic/paper_cgme_shadow_cosmetic.pdf", "Paper results: C_GME, Figs. 1-6",
  cgme_only(figures("shadow_cosmetic/paper", "paper"))),
]
for out, title, items in JOBS:
    write(out, title, items)
