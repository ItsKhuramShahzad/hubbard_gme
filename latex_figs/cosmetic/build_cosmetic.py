"""Cosmetic version of the paper C_GME figures (Figs. 1-6), full state.
Same data and panel sources as ../pdf_10pt/build (10 pt labels, 1.2 pt lines); changes:
  * the legend is replaced by a label "n = ..." written next to each curve, at the place where that curve is
    furthest from the others (chosen automatically, label boxes may not touch any curve or other label);
  * Fig. 4: dashed line at V = U/2 (U/t = 10) with the SDW and CDW sides marked, CDW side lightly shaded.
The data lines are copied unchanged. Run from this folder: ~/miniconda3/envs/qchem/bin/python build_cosmetic.py"""
import os, re, glob, shutil, subprocess, numpy as np, fitz
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), "pdf_10pt", "build")
WORK = os.path.join(HERE, "build"); os.makedirs(WORK, exist_ok=True)
TEX = os.path.expanduser("~/miniconda3/envs/texbuild/bin/tectonic")
AXW, AXH = 196.0, 135.0          # axis box in pt, measured on the compiled panels
LW, LH = 33.0, 10.0              # label box in pt (\normalsize "n = 3", 10 pt like the body text)

# labels placed by hand where the automatic choice is ambiguous: (panel file, n) -> (x, y, anchor)
OVERRIDE = {("fig4_p0.tex", 5): (1.5, 0.635, "south"), ("fig4_p0.tex", 6): (11.0, 0.975, "north"),
            ("fig4_p1.tex", 8): (8.0, 1.0, "west"), ("fig4_p1.tex", 3): (3.0, 0.192, "north"), ("fig2_p1.tex", 3): (25.0, 0.0, "north")}

SHIFT = {"south": "yshift=2.5pt", "north": "yshift=-2.5pt", "east": "xshift=-3pt", "west": "xshift=3pt"}

def opt(s, key):
    m = re.search(rf"\b{key}=([-0-9.]+)", s); return float(m.group(1)) if m else None

def curves(s):
    out = []
    # colour gmeC<k> is cluster size n = k + 2 in every panel (checked against the legend entries where present)
    for m in re.finditer(r"\\addplot\[(.*?)\] coordinates \{(.*?)\};(\s*\\addlegendentry\{\$n=(\d+)\$\})?", s, re.S):
        col = re.search(r"color=(\w+)", m.group(1)).group(1); n = int(col[4:]) + 2
        if m.group(4): assert int(m.group(4)) == n, (col, m.group(4))
        pts = np.array([(float(a), float(b)) for a, b in re.findall(r"\(([^,]+),([^)]+)\)", m.group(2))])
        out.append((n, col, pts))
    return out

def place(s, extra_obstacles=(), fname=""):
    cs = curves(s)
    xmin, xmax, ymin, ymax = (opt(s, k) for k in ("xmin", "xmax", "ymin", "ymax"))
    allp = np.vstack([c[2] for c in cs])
    xmin = allp[:, 0].min() if xmin is None else xmin; xmax = allp[:, 0].max() if xmax is None else xmax
    ymin = allp[:, 1].min() if ymin is None else ymin; ymax = allp[:, 1].max() if ymax is None else ymax
    T = lambda p: np.c_[(p[:, 0] - xmin) / (xmax - xmin) * AXW, (p[:, 1] - ymin) / (ymax - ymin) * AXH]
    dense = []
    for n, col, p in cs:                     # curve samples every ~0.5 pt, in pt
        q = T(p); seg = [q[0:1]]
        for a, b in zip(q[:-1], q[1:]):
            k = max(2, int(np.hypot(*(b - a)) / 0.5)); seg.append(np.linspace(a, b, k)[1:])
        dense.append(np.vstack(seg))
    def dist(box, P):
        x0, y0, x1, y1 = box
        dx = np.maximum(np.maximum(x0 - P[:, 0], 0), P[:, 0] - x1); dy = np.maximum(np.maximum(y0 - P[:, 1], 0), P[:, 1] - y1)
        return np.hypot(dx, dy).min()
    boxes = [tuple(b) for b in extra_obstacles]
    nodes = []
    # hardest curves first: the ones closest to another curve on average
    order = sorted(range(len(cs)), key=lambda i: np.mean([dist((*d, *d), dense[j]) for d in dense[i][::20]
                                                         for j in range(len(cs)) if j != i][:200] or [1e9]))
    inv = lambda c: ((c[0] / AXW) * (xmax - xmin) + xmin, (c[1] / AXH) * (ymax - ymin) + ymin)
    for i in order:
        if (fname, cs[i][0]) in OVERRIDE:
            x, y, a = OVERRIDE[(fname, cs[i][0])]; col = cs[i][1]
            nodes.append(rf"\node[font=\normalsize\rmfamily, text={col}!75!black, anchor={a}, {SHIFT[a]}, inner sep=0.5pt] "
                         rf"at (axis cs:{x:g},{y:g}) {{$n={cs[i][0]}$}};"); continue
        n, col, p = cs[i]; cand = np.vstack([dense[i][::6], dense[i][-1:]]); best = None; fallback = None
        for k in range(len(cand)):
            for side in ("south", "north", "east", "west"):
                cx, cy = cand[k]
                box = {"south": (cx - LW / 2, cy + 2.5, cx + LW / 2, cy + 2.5 + LH),
                       "north": (cx - LW / 2, cy - 2.5 - LH, cx + LW / 2, cy - 2.5),
                       "east": (cx - 3 - LW, cy - LH / 2, cx - 3, cy + LH / 2),
                       "west": (cx + 3, cy - LH / 2, cx + 3 + LW, cy + LH / 2)}[side]
                if box[0] < 2 or box[2] > AXW - 2 or box[1] < 2 or box[3] > AXH - 2: continue
                own = dist(box, dense[i]); oth = min([dist(box, dense[j]) for j in range(len(cs)) if j != i] or [99])
                lab = min([max(max(b[0] - box[2], box[0] - b[2]), max(b[1] - box[3], box[1] - b[3])) for b in boxes] or [99])
                # the label must sit clearly closer to its own curve than to any other one
                slack = min(own - 2.0, oth - 3.0, lab - 1.5, oth - own - 2.5)
                if fallback is None or slack > fallback[0]: fallback = (slack, k, side, box)
                if slack < 0: continue
                score = min(oth, 20) + 0.3 * min(lab, 20) + 1.0 * (k / len(cand))   # far from others, slight right preference
                if best is None or score > best[0]: best = (score, k, side, box)
        if best is None:
            print(f"   n={n}: no fully free place, closest gap {fallback[0]:.1f} pt"); best = fallback
        _, k, side, box = best; boxes.append(box)
        x, y = inv(cand[k])
        nodes.append(rf"\node[font=\normalsize\rmfamily, text={col}!75!black, anchor={side}, "
                     rf"{SHIFT[side]}, inner sep=0.5pt] at (axis cs:{x:.4g},{y:.4g}) {{$n={n}$}};")
    return nodes

def convert(src, dst, fig4=False):
    s = open(src).read()
    obstacles = []
    if fig4:
        # SDW/CDW words: fixed boxes, given in axis pt, kept free of labels
        xmin, xmax, ymin, ymax = (opt(s, k) for k in ("xmin", "xmax", "ymin", "ymax"))
        P = lambda x, y: ((x - xmin) / (xmax - xmin) * AXW, (y - ymin) / (ymax - ymin) * AXH)
        for xc in (2.5, 8.5):
            px, py = P(xc, 0.82); obstacles.append((px - 16, py - 6, px + 16, py + 6))
    if os.path.basename(src).startswith("fig2_p"):
        s = s.replace("ymin=-0.03,", "ymin=-0.12,")
    nodes = place(s, obstacles, os.path.basename(src))
    s = re.sub(r"\s*\\addlegendentry\{[^}]*\}", "", s)
    s = re.sub(r"\n\s*legend style=\{at=.*?\},\n", "\n", s)
    extra = ""
    if fig4:
        extra = (r"    \fill[gray!9] (axis cs:5,-0.03) rectangle (axis cs:12.24,1.05);" "\n"
                 r"    \draw[black!55, dashed, line width=0.6pt] (axis cs:5,-0.03) -- (axis cs:5,1.05);" "\n"
                 r"    \node[font=\normalsize\rmfamily, text=black!70] at (axis cs:2.5,0.82) {SDW};" "\n"
                 r"    \node[font=\normalsize\rmfamily, text=black!70] at (axis cs:8.5,0.82) {CDW};" "\n")
        s = s.replace("  ]\n    \\addplot", "  ]\n" + extra + "    \\addplot", 1)
    s = s.replace("  \\end{axis}", "    " + "\n    ".join(nodes) + "\n  \\end{axis}", 1)
    open(dst, "w").write(s)

def crop(src_pdf, dst_pdf):
    src = fitz.open(src_pdf); p = src[0]; bb = fitz.Rect()
    for d in p.get_drawings(): bb |= d["rect"]
    for b in p.get_text("dict")["blocks"]: bb |= fitz.Rect(b["bbox"])
    bb = (bb + (-2, -2, 2, 2)) & p.rect
    o = fitz.open(); q = o.new_page(width=bb.width, height=bb.height); q.show_pdf_page(q.rect, src, 0, clip=bb)
    o.save(dst_pdf, garbage=4, deflate=True)

# Figs. 3 and 6 are kept as they are (legend): in Fig. 3 all curves bunch at low T and drop together, in Fig. 6 the
# curves start at different K and cross, so there is no clean place for 10 pt labels next to the curves.
for name in ("fig1", "fig2", "fig4", "fig5"):
    shutil.copy(os.path.join(SRC, name + ".tex"), WORK)
    for p in sorted(glob.glob(os.path.join(SRC, name + "_p[0-9].tex"))):
        print(" ", os.path.basename(p))
        convert(p, os.path.join(WORK, os.path.basename(p)), fig4=(name == "fig4"))
    r = subprocess.run([TEX, name + ".tex"], cwd=WORK, capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr[-2000:])
    crop(os.path.join(WORK, name + ".pdf"), os.path.join(HERE, name + ".pdf")); print("built", name)
