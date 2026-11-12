"""Shadow figures with the cosmetic additions, plus one complete PDF of all of them.
Sources (left unchanged): ../cosmetic/build (paper C_GME Figs. 1, 2, 4, 5 with n labels next to the curves and the Fig. 4
transition marks), ../pdf_10pt/build (all other paper figures), ../projected (figures without doublons),
../fig12_combined (merged Fig. 1 + 2). Changes made here:
  * every Fig. 4 version (paper and without doublons): dashed line at V = U/2 = 5 (U/t = 10), CDW side lightly shaded,
    'SDW' / 'CDW' written where a free spot exists (checked against the curves; left out otherwise);
  * a soft shadow behind every data line (as in ../shadow).
Output: paper/, projected/, fig12_combined/ and all_figures_shadow_cosmetic.pdf in this folder.
Run: ~/miniconda3/envs/qchem/bin/python build_shadow_cosmetic.py"""
import os, re, glob, shutil, subprocess, numpy as np, fitz
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
TEX = os.path.expanduser("~/miniconda3/envs/texbuild/bin/tectonic")
SHADOW = ("line width=1.2pt, preaction={draw=black, draw opacity=0.22, line width=2.0pt, "
          "transform canvas={xshift=0.6pt, yshift=-0.6pt}}]")
AXW, AXH, VT = 196.0, 135.0, 5.0          # axis box in pt; transition at V/t = U/(2t) = 5

def opt(s, key):
    m = re.search(rf"\b{key}=([-0-9.]+)", s); return float(m.group(1)) if m else None

def add_transition(s):
    if r"\fill[gray!9]" in s:              # already marked (cosmetic Fig. 4 C_GME panels)
        return s
    xmin, xmax, ymin, ymax = (opt(s, k) for k in ("xmin", "xmax", "ymin", "ymax"))
    T = lambda x, y: ((x - xmin) / (xmax - xmin) * AXW, (y - ymin) / (ymax - ymin) * AXH)
    dense = []
    for c in re.findall(r"\\addplot\[.*?\] coordinates \{(.*?)\};", s, re.S):
        q = np.array([T(float(a), float(b)) for a, b in re.findall(r"\(([^,]+),([^)]+)\)", c)])
        for a, b in zip(q[:-1], q[1:]): dense.append(np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 0.5))))
        dense.append(q)
    P = np.vstack(dense)
    leg = re.search(r"legend style=\{at=\{\(([\d.]+),([\d.]+)\)\}", s)        # keep words away from the legend box
    lg = (float(leg.group(1)) * AXW, float(leg.group(2)) * AXH) if leg and r"\addlegendentry" in s else None
    words = []
    for word, xc in (("SDW", VT / 2), ("CDW", (VT + xmax) / 2)):
        px = T(xc, 0)[0]
        for yr in np.arange(0.90, 0.10, -0.02):
            py = yr * AXH; box = (px - 15, py - 6, px + 15, py + 6)
            dx = np.maximum(np.maximum(box[0] - P[:, 0], 0), P[:, 0] - box[2]); dy = np.maximum(np.maximum(box[1] - P[:, 1], 0), P[:, 1] - box[3])
            if np.hypot(dx, dy).min() < 3: continue
            if lg and abs(px - lg[0]) < 60 and abs(py - lg[1]) < 30: continue
            words.append(rf"    \node[font=\normalsize\rmfamily, text=black!70] at ({{axis cs:{xc:g},0}} |- {{rel axis cs:0,{yr:.2f}}}) {{{word}}};")
            break
    extra = (rf"    \fill[gray!9] ({{axis cs:{VT:g},0}} |- {{rel axis cs:0,0}}) rectangle ({{rel axis cs:1,1}});" "\n"
             rf"    \draw[black!55, dashed, line width=0.6pt] ({{axis cs:{VT:g},0}} |- {{rel axis cs:0,0}}) -- ({{axis cs:{VT:g},0}} |- {{rel axis cs:0,1}});" "\n"
             + "".join(w + "\n" for w in words))
    assert "  ]\n    \\addplot" in s
    return s.replace("  ]\n    \\addplot", "  ]\n" + extra + "    \\addplot", 1)

def add_shadow(s, path):
    assert "line width=1.2pt]" in s, path
    # with a 10 pt legend there is no reliable free spot inside the axes: put every legend above its plot, 3 columns
    if r"\addlegendentry" in s:
        s = re.sub(r"\n\s*legend style=\{at=\{\([\d.]+,[\d.]+\)\}, anchor=[a-z ]+, fill=white\},", "", s)
        s = re.sub(r"legend columns=\d+,", "legend columns=3,", s)
    s = s.replace(r"tick label style={font=\footnotesize\rmfamily", r"tick label style={font=\normalsize\rmfamily")   # ticks 10 pt
    s = s.replace(r"legend style={font=\scriptsize\rmfamily", r"legend style={font=\normalsize\rmfamily")         # legend 10 pt
    return s.replace("line width=1.2pt]", SHADOW)

def crop(src_pdf, dst_pdf):
    src = fitz.open(src_pdf); p = src[0]; bb = fitz.Rect()
    for d in p.get_drawings(): bb |= d["rect"]
    for b in p.get_text("dict")["blocks"]: bb |= fitz.Rect(b["bbox"])
    bb = (bb + (-2, -2, 2, 2)) & p.rect
    o = fitz.open(); q = o.new_page(width=bb.width, height=bb.height); q.show_pdf_page(q.rect, src, 0, clip=bb)
    o.save(dst_pdf, garbage=4, deflate=True)

def compile_all(work, wrappers, outdir):
    def one(w):
        r = subprocess.run([TEX, w], cwd=work, capture_output=True, text=True)
        if r.returncode: raise RuntimeError(w + "\n" + r.stderr[-1500:])
        crop(os.path.join(work, w[:-4] + ".pdf"), os.path.join(outdir, w[:-4] + ".pdf"))
    with ThreadPoolExecutor(6) as ex: list(ex.map(one, wrappers))
    print(os.path.basename(outdir), len(wrappers))

def prepare(work, files, is_panel, is_fig4):
    os.makedirs(work, exist_ok=True)
    for f in files: shutil.copy(f, work)
    for f in glob.glob(os.path.join(work, "*.tex")):
        b = os.path.basename(f)
        if not is_panel(b): continue
        s = open(f).read()
        if is_fig4(b): s = add_transition(s)
        open(f, "w").write(add_shadow(s, f))

# 1. paper figures: cosmetic C_GME Figs. 1, 2, 4, 5; everything else from pdf_10pt
out = os.path.join(HERE, "paper"); work = os.path.join(out, "build")
cos = {os.path.basename(f) for f in glob.glob(os.path.join(UP, "cosmetic", "build", "*.tex"))}
files = [os.path.join(UP, "cosmetic", "build", b) for b in cos] + \
        [f for f in glob.glob(os.path.join(UP, "pdf_10pt", "build", "*.tex")) if os.path.basename(f) not in cos]
prepare(work, files, lambda b: re.search(r"_p\d\.tex$", b) is not None, lambda b: b.startswith("fig4"))
compile_all(work, sorted(b for b in os.listdir(work) if b.endswith(".tex") and not re.search(r"_p\d\.tex$", b)), out)

# 2. figures without doublons
out = os.path.join(HERE, "projected"); work = os.path.join(out, "build")
prepare(work, glob.glob(os.path.join(UP, "projected", "fig*_proj*.tex")), lambda b: b.endswith("_panel.tex"),
        lambda b: b.startswith("fig4"))
compile_all(work, sorted(b for b in os.listdir(work) if b.endswith(".tex") and not b.endswith("_panel.tex")), out)

# 3. merged Fig. 1 + 2
out = os.path.join(HERE, "fig12_combined"); work = os.path.join(out, "build")
prepare(work, glob.glob(os.path.join(UP, "fig12_combined", "*.tex")), lambda b: b.startswith("panel_"), lambda b: False)
compile_all(work, ["fig12_combined.tex"], out)

# 4. one complete PDF: A4 landscape, figures at 100 % size, packed in rows, a short heading above each
QN = {"": "C_GME", "_purity": "purity", "_linear_entropy": "linear entropy", "_vN_entropy": "von Neumann entropy", "_omega": "detection probability omega_n"}
def heading(group, name):
    if group == "paper":
        m = re.match(r"fig(\d)(.*)", name); return f"Fig. {m.group(1)}: {QN[m.group(2)]}"
    if group == "projected":
        m = re.match(r"fig(\d)([ab]?)_proj(.*)", name)
        return f"Fig. {m.group(1)}{'(' + m.group(2) + ')' if m.group(2) else ''} without doublons: {QN[m.group(3)]}"
    return "Fig. 1 + Fig. 2 merged"
def order(name):
    m = re.match(r"fig(\d)([ab]?)(?:_proj)?(.*)", name)
    return (int(m.group(1)), m.group(2), list(QN).index(m.group(3)))
items = []
for group in ("paper", "fig12_combined", "projected"):
    names = [os.path.basename(f)[:-4] for f in glob.glob(os.path.join(HERE, group, "*.pdf"))]
    for n in sorted(names, key=lambda n: order(n) if n.startswith("fig") and n != "fig12_combined" else (0, "", 0)):
        items.append((group, n, heading(group, n)))
W, H, M, HEAD, GX, GY, TOP = 842, 595, 24, 16, 18, 10, 30
doc = fitz.open(); pg = None; x = y = rowh = 0; pages = 0
def new_page():
    global pg, x, y, rowh
    pg = doc.new_page(width=W, height=H); x, y, rowh = M, TOP, 0
    pg.insert_text((M, 18), "Figures with line shadows, transition marks and curve labels", fontsize=9.5)
for group, name, head in items:
    src = fitz.open(os.path.join(HERE, group, name + ".pdf")); r = src[0].rect; h = HEAD + r.height
    assert r.width <= W - 2 * M and h <= H - TOP - M, name
    if pg is None: new_page()
    if x + r.width > W - M: x, y, rowh = M, y + rowh + GY, 0
    if y + h > H - M: new_page()
    pg.insert_text((x, y + 10), head, fontsize=9)
    pg.show_pdf_page(fitz.Rect(x, y + HEAD, x + r.width, y + HEAD + r.height), src, 0)       # scale 1:1
    x += r.width + GX; rowh = max(rowh, h)
doc.save(os.path.join(HERE, "all_figures_shadow_cosmetic.pdf"), garbage=4, deflate=True)
print("all_figures_shadow_cosmetic.pdf:", len(doc), "pages,", len(items), "figures")
