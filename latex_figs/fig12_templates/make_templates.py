"""Two styles of the merged Fig. 1 + Fig. 2 (panels a-d), made from the same panel files in ../fig12_combined.
The data lines are copied unchanged; only axis options are edited. Run from this folder, then compile each
template with tectonic."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "fig12_combined"
PANELS = {"a": "panel_a_gs_L6", "b": "panel_b_es_L6", "c": "panel_c_gs_L8", "d": "panel_d_es_L8"}
GRID = "    grid=both,\n    grid style={line width=.1pt, draw=gray!20},\n"
LETTER = r"\node[anchor=east, font=\small\rmfamily] at (axis description cs:-0.13,0.954) {(%s)};"


def rep(text, old, new, count=None):
    n = text.count(old)
    assert n >= 1 and (count is None or n == count), (old[:60], n)
    return text.replace(old, new)


def standard(p, t):
    t = rep(t, GRID, "    tick align=inside,\n    tick pos=both,\n    minor tick num=1,\n", 1)
    t = rep(t, LETTER % p, r"\node[anchor=south west, font=\small\rmfamily, inner sep=1pt] at (axis description cs:0,1.02) {(%s)};" % p, 1)
    t = rep(t, "mark size=1.1pt, line width=0.7pt", "mark size=1.3pt, line width=0.8pt")
    return t


def advanced(p, t):
    # equal plot boxes in all four panels (pgfplots otherwise counts the hidden labels in the width)
    t = rep(t, "    width=\\columnwidth,\n    height=0.78\\columnwidth,\n",
            "    scale only axis,\n    width=0.82\\columnwidth,\n    height=0.6\\columnwidth,\n", 1)
    t = rep(t, GRID, "    grid=major,\n    grid style={line width=.2pt, draw=gray!30, dashed},\n    tick align=inside,\n", 1)
    x = "-0.13" if p in "ac" else "-0.03"           # right column has no y tick labels
    t = rep(t, LETTER % p, r"\node[anchor=east, font=\small\rmfamily\bfseries] at (axis description cs:%s,0.954) {(%s)};" % (x, p), 1)
    t = rep(t, "mark size=1.1pt, line width=0.7pt", "mark size=1.5pt, line width=0.9pt, mark options={fill=white}")
    t = t.replace("mark options={fill=white}, mark options={rotate=180}", "mark options={fill=white, rotate=180}")
    if p in "ab":                                   # top row: column titles, x labels only at the bottom
        title = "Ground state" if p == "a" else "First excited singlet"
        t = rep(t, "    xlabel={$U/t$},\n", "    title={%s},\n    title style={font=\\small\\rmfamily, yshift=-2pt},\n    xticklabels={},\n" % title, 1)
    if p in "bd":                                   # right column: y label only on the left, row label on the right
        t = rep(t, "    ylabel={$C_{\\mathrm{GME}}$},\n", "    yticklabels={},\n", 1)
        L = "6" if p == "b" else "8"
        t = rep(t, "  ]\n", "    after end axis/.append code={\\node[rotate=-90, font=\\small\\rmfamily] at (axis description cs:1.05,0.5) {$L=%s$};},\n  ]\n" % L, 1)
    if p == "d":                                    # one legend for the whole figure, placed above it
        t = rep(t, "    legend style={at={(0.620,0.250)}, anchor=center, fill=white},\n", "    legend to name=fig12legend,\n    legend style={draw=none, font=\\footnotesize\\rmfamily},\n", 1)
        t = rep(t, "    legend columns=3,\n", "    legend columns=6,\n", 1)
    return t


WRAP = r"""\documentclass[border=4pt]{standalone}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\begin{document}
\setlength{\columnwidth}{8.6cm}%%
\pgfplotsset{every axis/.append style={ytick={0,0.2,0.4,0.6,0.8,1}}}%%
\begin{tabular}{@{}c@{\hspace{%s}}c@{}}
%s\input{panel_a.tex} & \input{panel_b.tex}\\[%s]
\input{panel_c.tex} & \input{panel_d.tex}
\end{tabular}
\end{document}
"""

for name, fn, gap, vgap, top in (("standard", standard, "6mm", "2pt", ""),
                                 ("advanced", advanced, "-8mm", "-2pt", "\\multicolumn{2}{c}{\\ref{fig12legend}}\\\\[2pt]\n")):
    out = Path(__file__).resolve().parent / name
    for p, f in PANELS.items():
        (out / f"panel_{p}.tex").write_text(fn(p, (SRC / f"{f}.tex").read_text()))
    colors = "".join(l + "\n" for l in (SRC / "panel_a_gs_L6.tex").read_text().splitlines() if l.startswith("\\definecolor"))
    (out / f"fig12_{name}.tex").write_text((WRAP % (gap, top, vgap)).replace("\\begin{document}", colors + "\\begin{document}"))
print("written")
