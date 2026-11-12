"""C_GME of the ground state against n, one line for each of the 20 U/t values of Fig. 1 (U/t = 1..20),
(a) L = 6, (b) L = 8, half filling (K = L), V = 0, t = 1. Colour runs from light (U/t = 1) to dark (U/t = 20).
Values are read directly from the Fig. 1 PKLs (pkl_values.py: result['C_GME'], md5 checked against FIGURES_DATA): uscan_v1_L6_Ne6_t1_..._sz0_restarts10_iter200_kmax4_..._U1to20_N20-d03a74da789e.pkl and
uscan_v1_L8_Ne8_t1_..._full_restarts8_iter200_kmax4_..._U1to20_N20-d03a74da789e.pkl.
Run from this folder: ~/miniconda3/envs/qchem/bin/python build_cgme_vs_n_allU.py"""
import os, re, subprocess, numpy as np, fitz
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), "pdf_10pt", "build")
TEX = os.path.expanduser("~/miniconda3/envs/texbuild/bin/tectonic")
STOPS = [(158, 202, 225), (66, 146, 198), (8, 81, 156), (8, 38, 84)]      # light blue -> dark navy

import sys
sys.path.insert(0, HERE)
from pkl_values import cgme

def colour(f):                                   # f in [0, 1]
    x = f * (len(STOPS) - 1); i = min(int(x), len(STOPS) - 2); w = x - i
    return tuple(int(round((1 - w) * a + w * b)) for a, b in zip(STOPS[i], STOPS[i + 1]))

Us = list(range(1, 21)); defs = []; data_lines = []
for k, U in enumerate(Us):
    r, g, b = colour(k / (len(Us) - 1)); defs.append(rf"\definecolor{{cU{U}}}{{RGB}}{{{r},{g},{b}}}")

def panel(L, letter):
    name, t = cgme(L); ns = list(t)
    for U in Us: assert all(float(U) in t[n] for n in ns), (L, U)
    data_lines.append(f"L={L}: PKL {name}\n  n = {ns}")
    plots = []
    for U in Us:
        vals = [t[n][float(U)] for n in ns]
        data_lines.append(f"  U/t={U:2d}: " + "  ".join(f"{v:.6f}" for v in vals))
        pts = " ".join(f"({n},{v:g})" for n, v in zip(ns, vals))
        plots.append(rf"    \addplot[color=cU{U}, mark=*, mark size=1.1pt, line width=1.2pt] coordinates {{{pts}}};")
    cmap = " ".join(f"rgb255=({r},{g},{b})" for r, g, b in STOPS)
    return (r"""\begin{tikzpicture}
  \begin{axis}[
    width=\columnwidth, height=0.78\columnwidth,
    xlabel={number of spins $n$}, ylabel={$C_{\mathrm{GME}}$},
    xmin=2.7, xmax=""" + f"{L + 0.3}" + r""", ymin=-0.03, ymax=1.05, xtick={""" + ",".join(map(str, ns)) + r"""},
    grid=both, grid style={line width=.1pt, draw=gray!20},
    tick label style={font=\footnotesize\rmfamily, /pgf/number format/fixed},
    label style={font=\normalsize\rmfamily},
    colormap={uramp}{""" + cmap + r"""},
    colorbar, point meta min=1, point meta max=20,
    colorbar style={width=2.5mm, ytick={1,5,10,15,20}, ylabel={$U/t$},
                    tick label style={font=\footnotesize\rmfamily}, ylabel style={font=\normalsize\rmfamily}},
    after end axis/.append code={\node[anchor=east, font=\normalsize\rmfamily] at (axis description cs:-0.13,0.954) {""" + letter + r"""};},
  ]
""" + "\n".join(plots) + r"""
  \end{axis}
\end{tikzpicture}""")

body = panel(6, "(a)") + r"\\[2pt]" + "\n" + panel(8, "(b)")
tex = (r"\documentclass[border=4pt]{standalone}" "\n" r"\usepackage{pgfplots}\pgfplotsset{compat=1.18}" "\n" +
       "\n".join(defs) + "\n" r"\begin{document}" "\n" r"\setlength{\columnwidth}{8.6cm}%" "\n"
       r"\begin{tabular}{@{}c@{}}" "\n" + body + "\n" r"\end{tabular}" "\n" r"\end{document}" "\n")
open(os.path.join(HERE, "cgme_vs_n_allU.tex"), "w").write(tex)
open(os.path.join(HERE, "cgme_vs_n_allU_data.txt"), "w").write(
    "C_GME (ground state, K=L, V=0, t=1) read directly from the PKLs; columns = n\n" + "\n".join(data_lines) + "\n")
r = subprocess.run([TEX, "cgme_vs_n_allU.tex"], cwd=HERE, capture_output=True, text=True)
if r.returncode: raise SystemExit(r.stderr[-2500:])
src = fitz.open(os.path.join(HERE, "cgme_vs_n_allU.pdf")); p = src[0]; bb = fitz.Rect()
for d in p.get_drawings(): bb |= d["rect"]
for b in p.get_text("dict")["blocks"]: bb |= fitz.Rect(b["bbox"])
bb = (bb + (-2, -2, 2, 2)) & p.rect
o = fitz.open(); q = o.new_page(width=bb.width, height=bb.height); q.show_pdf_page(q.rect, src, 0, clip=bb)
src.close(); tmp = os.path.join(HERE, "_c.pdf"); o.save(tmp); os.replace(tmp, os.path.join(HERE, "cgme_vs_n_allU.pdf"))
print("built", round(bb.width), round(bb.height))
