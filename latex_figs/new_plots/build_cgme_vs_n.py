"""C_GME of the ground state against the number of spins n, at fixed U/t (10 and 20), for L = 6 and L = 8
(K = L, V = 0, t = 1). The values are read directly from the Fig. 1 PKLs (pkl_values.py: result['C_GME'],
md5 checked against FIGURES_DATA).
Run from this folder: ~/miniconda3/envs/qchem/bin/python build_cgme_vs_n.py"""
import os, re, subprocess, fitz
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), "pdf_10pt", "build")
TEX = os.path.expanduser("~/miniconda3/envs/texbuild/bin/tectonic")

import sys
sys.path.insert(0, HERE)
from pkl_values import cgme

src = {L: cgme(L) for L in (6, 8)}
data = {(L, U): {n: src[L][1][n][float(U)] for n in src[L][1]} for L in (6, 8) for U in (10, 20)}
with open(os.path.join(HERE, "cgme_vs_n_data.txt"), "w") as fh:
    fh.write("C_GME (ground state, K=L, V=0, t=1) read from the PKLs " + ", ".join(src[L][0] for L in (6, 8)) + "\n")
    for (L, U), d in data.items():
        fh.write(f"L={L} U/t={U}: " + "  ".join(f"n={n}: {v:.6f}" for n, v in d.items()) + "\n")

col = {6: "gmeC0", 8: "gmeC3"}; mark = {6: "*", 8: "square*"}; dash = {10: "solid", 20: "dashed"}
plots = []
for (L, U), d in data.items():
    pts = " ".join(f"({n},{v:g})" for n, v in d.items())
    plots.append(rf"    \addplot[color={col[L]}, mark={mark[L]}, mark size=1.6pt, line width=1.2pt, {dash[U]}] coordinates {{{pts}}};")
    if L == 6: plots.append(rf"    \addlegendentry{{$U/t={U}$}}")
# direct labels for L: L=6 above-left of its curves, L=8 below the start of its curves
lab = (rf"    \node[font=\normalsize\rmfamily, text=gmeC0!75!black, anchor=south east] at (axis cs:4.5,0.66) {{$L=6$}};" "\n"
       rf"    \node[font=\normalsize\rmfamily, text=gmeC3!75!black, anchor=north west] at (axis cs:3,{min(data[8,10][3], data[8,20][3]):g}) {{$L=8$}};")
tex = r"""\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\definecolor{gmeC0}{HTML}{1F77B4}
\definecolor{gmeC3}{HTML}{D62728}
\begin{document}
\setlength{\columnwidth}{8.6cm}%
\begin{tikzpicture}
  \begin{axis}[
    width=\columnwidth, height=0.78\columnwidth,
    xlabel={number of spins $n$}, ylabel={$C_{\mathrm{GME}}$},
    xmin=2.6, xmax=8.4, ymin=0, ymax=1.05, xtick={3,4,5,6,7,8},
    grid=both, grid style={line width=.1pt, draw=gray!20},
    tick label style={font=\footnotesize\rmfamily, /pgf/number format/fixed},
    label style={font=\normalsize\rmfamily},
    legend style={font=\normalsize\rmfamily, draw=gray!40, fill=white, at={(0.97,0.04)}, anchor=south east},
    legend image post style={black}, legend cell align={left},
  ]
""" + "\n".join(plots) + "\n" + lab + r"""
  \end{axis}
\end{tikzpicture}
\end{document}
"""
open(os.path.join(HERE, "cgme_vs_n.tex"), "w").write(tex)
r = subprocess.run([TEX, "cgme_vs_n.tex"], cwd=HERE, capture_output=True, text=True)
if r.returncode: raise SystemExit(r.stderr[-2000:])
src = fitz.open(os.path.join(HERE, "cgme_vs_n.pdf")); p = src[0]; bb = fitz.Rect()
for d in p.get_drawings(): bb |= d["rect"]
for b in p.get_text("dict")["blocks"]: bb |= fitz.Rect(b["bbox"])
bb = (bb + (-2, -2, 2, 2)) & p.rect
o = fitz.open(); q = o.new_page(width=bb.width, height=bb.height); q.show_pdf_page(q.rect, src, 0, clip=bb)
src.close(); o.save(os.path.join(HERE, "cgme_vs_n_c.pdf")); os.replace(os.path.join(HERE, "cgme_vs_n_c.pdf"), os.path.join(HERE, "cgme_vs_n.pdf"))
print(open(os.path.join(HERE, "cgme_vs_n_data.txt")).read())
